"""Recover a stale macOS VPN source address without changing system routes."""
from __future__ import annotations

import ipaddress
import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
from uuid import uuid4
from urllib.parse import urlsplit
from urllib.request import getproxies

import httpx
import websocket


class RoutedTransport(httpx.HTTPTransport):
    def __init__(self, **options):
        super().__init__(**options)
        self.options = options
        self.recovery = {}
        self.lock = threading.Lock()
        self.closed = threading.Event()

    def handle_request(self, request):
        began = time.monotonic()
        try:
            return super().handle_request(request)
        except httpx.ConnectError as exc:
            if self.closed.is_set():
                raise
            if request.method not in {"GET", "HEAD"}:
                # Keep the pre-existing connect-only errno49 recovery used by
                # ordinary HTTP POST callers. Do not add errno47 or replay a
                # write/read failure, and leave the original request untouched.
                if "[Errno 49]" not in str(exc):
                    raise
                address = routed_address(request.url.host)
                if not address or self.closed.is_set():
                    raise
                with self.lock:
                    if self.closed.is_set():
                        raise
                    if address not in self.recovery:
                        self.recovery[address] = httpx.HTTPTransport(local_address=address, **self.options)
                    transport = self.recovery[address]
                if self.closed.is_set():
                    raise
                return transport.handle_request(request)
            if not re.search(r"\[Errno (?:47|49)\]", str(exc)):
                raise
            timeout = request.extensions.get("timeout", {}).get("connect")
            deadline = began + timeout if isinstance(timeout, (int, float)) else None
            route = _public_route(request.url, deadline)
            if not route or self.closed.is_set():
                raise
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise
            routed = _model_request(request, route)
            if remaining is not None:
                routed.extensions["timeout"] = dict(request.extensions.get("timeout", {}), connect=remaining)
            with self.lock:
                if self.closed.is_set():
                    raise
                if route not in self.recovery:
                    self.recovery[route] = httpx.HTTPTransport(local_address=route[2], retries=0, **self.options)
                transport = self.recovery[route]
            if self.closed.is_set():
                raise
            return transport.handle_request(routed)

    def close(self):
        self.closed.set()
        super().close()
        with self.lock:
            transports = list(self.recovery.values())
        for transport in transports:
            transport.close()


def client(**kwargs):
    # Preserve user-configured HTTP proxies and the ordinary Windows path.
    if (sys.platform == "darwin" and not kwargs.get("proxy")
            and "transport" not in kwargs and "mounts" not in kwargs
            and not any(os.environ.get(k) for k in _PROXIES)):
        proxies = getproxies() if kwargs.get("trust_env", True) else {}
        if not any(proxies.get(k) for k in ("http", "https", "all")):
            options = {k: kwargs[k] for k in ("verify", "cert", "trust_env", "http1", "http2", "limits") if k in kwargs}
            kwargs["transport"] = RoutedTransport(**options)
    return httpx.Client(**kwargs)


def routed_address(host, *, timeout=3):
    if sys.platform != "darwin" or not host or host.startswith("-"):
        return None
    try:
        route = subprocess.run(["/sbin/route", "-n", "get", host], capture_output=True,
                               text=True, timeout=timeout, check=True).stdout
        interface = re.search(r"^\s*interface:\s*(en\d+)\s*$", route, re.M)
        if not interface:
            return None
        address = subprocess.run(["/usr/sbin/ipconfig", "getifaddr", interface[1]], capture_output=True,
                                 text=True, timeout=timeout, check=True).stdout.strip()
        return str(ipaddress.ip_address(address))
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


_PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_DNS_LOCK = threading.Lock()
_DNS_PENDING = {}


def _model_dns(host, port, timeout):
    # getaddrinfo cannot be interrupted portably. Bound the caller's wait and
    # allow at most one daemon lookup, shared by callers for this same origin.
    key = (host, port)
    with _DNS_LOCK:
        pending = _DNS_PENDING.get(key)
        if pending is None:
            if _DNS_PENDING:
                return []
            pending = {"done": threading.Event(), "answers": []}
            _DNS_PENDING[key] = pending
            resolve = socket.getaddrinfo

            def lookup():
                try:
                    pending["answers"] = resolve(host, port, socket.AF_INET, socket.SOCK_STREAM)
                except OSError:
                    pass
                finally:
                    with _DNS_LOCK:
                        _DNS_PENDING.pop(key, None)
                    pending["done"].set()

            threading.Thread(target=lookup, name="model-route-dns", daemon=True).start()
    if not pending["done"].wait(max(0, timeout)):
        return []
    return pending["answers"]


def _public_route(url, connect_deadline):
    """Recover a GET/HEAD connect using a same-family current DNS destination."""
    if sys.platform != "darwin" or url.scheme not in {"http", "https"}:
        return None
    deadline = min(time.monotonic() + 1.5, connect_deadline) if connect_deadline is not None else time.monotonic() + 1.5
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    answers = _model_dns(url.host, url.port or (443 if url.scheme == "https" else 80), min(.5, remaining))
    seen = set()
    for answer in answers:
        ip = answer[4][0]
        if answer[0] != socket.AF_INET or ip in seen:
            continue
        seen.add(ip)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or len(seen) > 8:
            break
        local = routed_address(ip, timeout=min(.35, remaining / 2))
        try:
            if local and ipaddress.ip_address(ip).version == ipaddress.ip_address(local).version == 4:
                return (url.scheme, url.host, url.port), ip, local
        except ValueError:
            continue
    return None


def _model_route(url, options):
    """Select a current DNS address on an en interface before any POST."""
    target = httpx.URL(url)
    if (sys.platform != "darwin" or target.scheme != "https" or options.get("proxy")
            or "transport" in options or any(os.environ.get(key) for key in _PROXIES)):
        return None
    # HTTPX also reads macOS system proxies. A custom transport must not disable
    # that configured path merely because no proxy environment variable exists.
    proxies = getproxies() if options.get("trust_env", True) else {}
    if any(proxies.get(key) for key in ("http", "https", "all")):
        return None
    allowance = 1.5
    timeout = options.get("timeout")
    if isinstance(timeout, httpx.Timeout):
        timeout = timeout.connect
    if isinstance(timeout, (int, float)):
        allowance = min(allowance, max(0, timeout))
    deadline = time.monotonic() + allowance
    answers = _model_dns(target.host, target.port or 443, min(.5, allowance))
    seen = set()
    for answer in answers:
        ip = answer[4][0]
        if ip in seen:
            continue
        seen.add(ip)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or len(seen) > 8:
            break
        local = routed_address(ip, timeout=min(.35, remaining / 2))
        if local:
            return (target.scheme, target.host, target.port), ip, local
    return None


def _model_request(request, route):
    if route is None:
        return request
    origin, ip, _ = route
    if (request.url.scheme, request.url.host, request.url.port) != origin:
        raise ValueError("模型连接目标已改变，未沿用原地址绑定")
    # Clone only the network destination. The original request, Host, payload,
    # charge/cache endpoint and TLS identity remain the configured hostname.
    extensions = dict(request.extensions, sni_hostname=request.url.host)
    return httpx.Request(request.method, request.url.copy_with(host=ip),
                         headers=request.headers, stream=request.stream, extensions=extensions)


class _ModelTransport(httpx.HTTPTransport):
    def __init__(self, route, **options):
        self.route = route
        super().__init__(local_address=route[2], retries=0, **options)

    def handle_request(self, request):
        return super().handle_request(_model_request(request, self.route))


class _AsyncModelTransport(httpx.AsyncHTTPTransport):
    def __init__(self, route, **options):
        self.route = route
        super().__init__(local_address=route[2], retries=0, **options)

    async def handle_async_request(self, request):
        return await super().handle_async_request(_model_request(request, self.route))


def _model_client_options(url, options, transport):
    options = dict(options)
    route = _model_route(url, options)
    if route:
        tls = {key: options[key] for key in ("verify", "cert", "trust_env") if key in options}
        options["transport"] = transport(route, **tls)
    return options


def model_client(url, **options):
    return httpx.Client(**_model_client_options(url, options, _ModelTransport))


@asynccontextmanager
async def model_async_client(url, **options):
    # Keep the bounded DNS/route probe off the event loop, so cancellation and
    # the caller's absolute deadline remain observable during preselection.
    configured = await asyncio.to_thread(_model_client_options, url, options, _AsyncModelTransport)
    async with httpx.AsyncClient(**configured) as connection:
        yield connection


class ModelTrace:
    """Failure-only phase timing; never retain request/response data or URLs."""
    PHASES = {"connection.connect_tcp", "connection.start_tls", "proxy.start_tls",
              "http11.send_request_headers", "http11.send_request_body", "http11.receive_response_headers",
              "http11.receive_response_body", "http11.response_closed"}

    def __init__(self):
        self.started = time.monotonic()
        self.events = []

    def record(self, name, info):
        phase, _, event = name.rpartition(".")
        if phase in self.PHASES and event in {"started", "complete", "failed"} and len(self.events) < 80:
            error = info.get("exception")
            self.events.append({"stage": name, "elapsed_seconds": round(time.monotonic()-self.started, 3),
                                "exception_type": type(error).__name__ if isinstance(error, BaseException) else None})

    async def async_record(self, name, info):
        self.record(name, info)

    def save(self, error):
        from . import paths
        target = paths.DATA / "diagnostics" / "model-transport" / (uuid4().hex + ".json")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as output:
                json.dump({"elapsed_seconds": round(time.monotonic()-self.started, 3),
                           "exception_type": type(error).__name__, "events": self.events}, output)
            error.transport_diagnostic_path = str(target)
        except OSError:
            pass


def post(url, **kwargs):
    # This function serves model POSTs only. Choose before sending; never resend
    # an ambiguous body after a write/read failure or change endpoints on error.
    trace = ModelTrace()
    options = {key: kwargs.pop(key) for key in ("verify", "cert", "proxy", "trust_env") if key in kwargs}
    if "timeout" in kwargs:
        options["timeout"] = kwargs["timeout"]
    extensions = dict(kwargs.pop("extensions", {}), trace=trace.record)
    try:
        with model_client(url, **options) as client:
            return client.post(url, extensions=extensions, **kwargs)
    except httpx.RequestError as exc:
        trace.save(exc)
        raise


def websocket_connection(url, *, timeout=None, **options):
    """Recover only a direct TCP connect failure, before TLS/WS or START."""
    parsed = urlsplit(url)
    proxies = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    if (sys.platform != "darwin" or parsed.scheme not in ("ws", "wss") or not parsed.hostname
            or "socket" in options or options.get("sslopt") or options.get("sockopt")
            or "class_" in options or os.environ.get("WEBSOCKET_CLIENT_CA_BUNDLE")
            or any(os.environ.get(key) for key in proxies)
            or any(value for key, value in options.items() if key.startswith("http_proxy_") or key == "proxy_type")):
        return websocket.create_connection(url, timeout=timeout, **options)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    effective_timeout = timeout if timeout is not None else websocket.getdefaulttimeout()
    sock = None
    try:
        try:
            sock = socket.create_connection((host, port), timeout=effective_timeout)
        except OSError as exc:
            if exc.errno != 49:
                raise
            address = routed_address(host)
            if not address:
                raise
            sock = socket.create_connection((host, port), timeout=effective_timeout,
                                            source_address=(address, 0))
        if parsed.scheme == "wss":
            # websocket-client skips TLS for supplied sockets. Preserve both
            # certificate/hostname verification and SNI before handing it over.
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        return websocket.create_connection(url, timeout=timeout, socket=sock, **options)
    except BaseException:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        raise
