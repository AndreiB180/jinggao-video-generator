// jinggao-assist — 远程协助 for 镜稿: lets the developer open this computer's 镜稿 page for a while.
//
//	jinggao-assist -app 127.0.0.1:8770 -config assist.json -key data/assist/assist.key
//
// It logs in to a relay server over SSH as a forward-only user and asks the relay to listen on one
// of its own loopback ports; every connection there is passed to the 镜稿 page on this computer.
// Nothing else on this computer is reachable, and on the relay only its administrator can reach that
// loopback port. assist.json (made by the package build, not in the repository) holds no secret:
//
//	{"relay": "host", "ports": [443, 22], "user": "…", "host_key": "ssh-ed25519 …", "listen": [23001, 23020]}
//
// The key is this computer's own: made on first use (-key), never shipped. Until the developer adds its
// public key on the relay, logins are refused and the program waits, trying again every 15 seconds.
//
// Status lines on stdout, read by the app: "KEY <public key line>", "WAIT <reason>", "LISTEN <port> <route>",
// "DOWN <reason>", "FAIL <reason>". It ends when stdin closes (the app quit or turned it off) or after -hours.
package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

type config struct {
	Relay   string `json:"relay"`
	Ports   []int  `json:"ports"`
	User    string `json:"user"`
	HostKey string `json:"host_key"`
	Listen  [2]int `json:"listen"`
}

var out = struct {
	sync.Mutex
	w io.Writer
}{w: os.Stdout}

func say(format string, a ...any) {
	out.Lock()
	defer out.Unlock()
	fmt.Fprintf(out.w, format+"\n", a...)
}

func main() {
	app := flag.String("app", "127.0.0.1:8770", "the 镜稿 page on this computer")
	path := flag.String("config", "assist.json", "relay settings")
	keyPath := flag.String("key", "assist.key", "this computer's key (made on first use)")
	hours := flag.Float64("hours", 12, "turn itself off after this long")
	flag.Parse()
	cfg, hostKey, err := load(*path)
	if err != nil {
		say("FAIL 远程协助的配置文件有问题：%v", err)
		os.Exit(2)
	}
	signer, err := ownKey(*keyPath)
	if err != nil {
		say("FAIL 无法生成这台电脑的协助钥匙：%v", err)
		os.Exit(2)
	}
	say("KEY %s", strings.TrimSpace(string(ssh.MarshalAuthorizedKey(signer.PublicKey())))+" jinggao-assist")
	go func() { // the app closes stdin when it quits or turns 远程协助 off
		io.Copy(io.Discard, os.Stdin)
		os.Exit(0)
	}()
	time.AfterFunc(time.Duration(*hours*float64(time.Hour)), func() {
		say("FAIL 已开了 %g 小时，自动关闭", *hours)
		os.Exit(0)
	})
	run(cfg, signer, hostKey, *app)
}

func load(path string) (config, ssh.PublicKey, error) {
	var cfg config
	b, err := os.ReadFile(path)
	if err != nil {
		return cfg, nil, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, nil, err
	}
	if cfg.Relay == "" || cfg.User == "" || len(cfg.Ports) == 0 || cfg.Listen[0] <= 0 || cfg.Listen[1] < cfg.Listen[0] {
		return cfg, nil, errors.New("relay/user/ports/listen 没有填全")
	}
	hostKey, _, _, _, err := ssh.ParseAuthorizedKey([]byte(cfg.HostKey))
	if err != nil {
		return cfg, nil, fmt.Errorf("host_key: %v", err)
	}
	return cfg, hostKey, nil
}

// ownKey reads this computer's key, making it the first time (ed25519, readable only by this user).
func ownKey(path string) (ssh.Signer, error) {
	if data, err := os.ReadFile(path); err == nil {
		return ssh.ParsePrivateKey(data)
	}
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	block, err := ssh.MarshalPrivateKey(private, "jinggao-assist")
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, pem.EncodeToMemory(block), 0o600); err != nil {
		return nil, err
	}
	return ssh.NewSignerFromKey(private)
}

// run keeps the relay listening until the program ends: it reconnects after a drop and asks for the
// port it had before first, so the number the developer was told stays the same.
func run(cfg config, signer ssh.Signer, hostKey ssh.PublicKey, app string) {
	wait, last := time.Second, 0
	for {
		client, route, err := connect(cfg, signer, hostKey)
		if err != nil {
			if strings.Contains(err.Error(), "unable to authenticate") {
				say("WAIT 等技术人员把这台电脑加进白名单，加好后会自动连上")
				time.Sleep(15 * time.Second)
				continue
			}
			say("DOWN 连不上中转服务器（%v），%d 秒后重试", short(err), int(wait.Seconds()))
			time.Sleep(wait)
			wait = min(wait*2, 30*time.Second)
			continue
		}
		listener, port := listen(client, cfg.Listen, last)
		if listener == nil {
			client.Close()
			say("DOWN 中转服务器上的协助端口都被占用，30 秒后重试")
			time.Sleep(30 * time.Second)
			continue
		}
		wait, last = time.Second, port
		say("LISTEN %d %s", port, route)
		go keepAlive(client)
		err = serve(listener, app)
		client.Close()
		say("DOWN 连接断开（%v），重新连接", short(err))
		time.Sleep(3 * time.Second)
	}
}

// connect tries each relay port in turn (443 first: networks that slow or block 22 rarely touch 443).
func connect(cfg config, signer ssh.Signer, hostKey ssh.PublicKey) (*ssh.Client, string, error) {
	clientConfig := &ssh.ClientConfig{
		User:            cfg.User,
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(signer)},
		HostKeyCallback: ssh.FixedHostKey(hostKey),
		// the relay also has RSA/ECDSA keys: ask for the type that is pinned
		HostKeyAlgorithms: []string{hostKey.Type()},
		Timeout:           15 * time.Second,
	}
	var errs []string
	for _, port := range cfg.Ports {
		addr := net.JoinHostPort(cfg.Relay, strconv.Itoa(port))
		conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%d: %v", port, err))
			continue
		}
		if tcp, ok := conn.(*net.TCPConn); ok {
			tcp.SetKeepAlive(true)
			tcp.SetKeepAlivePeriod(15 * time.Second)
		}
		conn.SetDeadline(time.Now().Add(20 * time.Second))
		c, chans, reqs, err := ssh.NewClientConn(conn, addr, clientConfig)
		if err != nil {
			conn.Close()
			if strings.Contains(err.Error(), "unable to authenticate") {
				return nil, "", err
			}
			errs = append(errs, fmt.Sprintf("%d: %v", port, err))
			continue
		}
		conn.SetDeadline(time.Time{})
		return ssh.NewClient(c, chans, reqs), fmt.Sprintf("port%d", port), nil
	}
	return nil, "", errors.New(strings.Join(errs, "; "))
}

// listen asks the relay for one loopback port in the range, the previous one first.
func listen(client *ssh.Client, span [2]int, previous int) (net.Listener, int) {
	ports := []int{}
	if previous > 0 {
		ports = append(ports, previous)
	}
	for p := span[0]; p <= span[1]; p++ {
		if p != previous {
			ports = append(ports, p)
		}
	}
	for _, p := range ports {
		if l, err := client.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(p))); err == nil {
			return l, p
		}
	}
	return nil, 0
}

// keepAlive closes the client when the relay stops answering, so run() reconnects instead of hanging.
func keepAlive(client *ssh.Client) {
	for {
		time.Sleep(15 * time.Second)
		answered := make(chan error, 1)
		go func() {
			_, _, err := client.SendRequest("keepalive@openssh.com", true, nil)
			answered <- err
		}()
		select {
		case err := <-answered:
			if err != nil {
				client.Close()
				return
			}
		case <-time.After(30 * time.Second):
			client.Close()
			return
		}
	}
}

// serve passes every connection the relay accepts to the 镜稿 page on this computer.
func serve(listener net.Listener, app string) error {
	for {
		remote, err := listener.Accept()
		if err != nil {
			return err
		}
		go func() {
			defer remote.Close()
			local, err := net.DialTimeout("tcp", app, 5*time.Second)
			if err != nil {
				return
			}
			defer local.Close()
			done := make(chan struct{}, 2)
			go func() { io.Copy(local, remote); done <- struct{}{} }()
			go func() { io.Copy(remote, local); done <- struct{}{} }()
			<-done
		}()
	}
}

func short(err error) string {
	if err == nil {
		return "已关闭"
	}
	text := err.Error()
	if len(text) > 160 {
		text = text[:160] + "…"
	}
	return text
}
