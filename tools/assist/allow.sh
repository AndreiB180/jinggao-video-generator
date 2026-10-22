#!/bin/bash
# Allow one customer computer's 远程协助 key on the relay (run on the developer's Mac).
#
#   tools/assist/allow.sh RELAY "ssh-ed25519 AAAA… jinggao-assist" 客户名
#
# RELAY is an ssh alias with root on the relay. The key (the 电脑识别码 the customer copied from
# 设置 → 合成与保存 → 远程协助) may only listen on the relay's 127.0.0.1:23001-23020, as the sshd Match
# block for the forward-only user also says; nothing else. Remove a line from
# /home/jgsupport/.ssh/authorized_keys to take a computer off again.
set -euo pipefail
relay=$1 key=$2 name=${3:-customer}
if [[ ! $key =~ ^ssh-ed25519\ [A-Za-z0-9+/]+=*([[:space:]]|$) ]]; then
	echo "这不是 ed25519 公钥：$key" >&2
	exit 2
fi
blob=$(awk '{print $1" "$2}' <<<"$key")
name=$(tr -c 'A-Za-z0-9_\n-' '_' <<<"$name")
perm=""
for p in $(seq 23001 23020); do perm+="permitlisten=\"127.0.0.1:$p\","; done
line="restrict,port-forwarding,${perm%,} $blob $name"
ssh -n "$relay" "f=/home/jgsupport/.ssh/authorized_keys; grep -qF '$blob' \$f || echo '$line' >> \$f; echo \"\$(grep -c . \$f) 台电脑已允许\""
