"""Allowed publisher and stock-library hosts for public material requests.

The historical module name is retained; stock hosts are deliberately exact so
adding a provider does not open arbitrary third-party or shared CDN tenants.
"""
from urllib.parse import urlparse

DOMAINS = (
    "cctv.com", "cntv.cn", "cctv.cn", "cctvpic.com", "cctvcdn.cn",
    "newcntv.qcloudcdn.com", "hls.cntv.lxdns.com",
    "chinanews.com.cn", "chinanews.com", "cns.com.cn", "cnsimg.net", "cnsphoto.com",
    "news.cn", "xinhuanet.com", "xinhuaxmt.com",
    "people.com.cn", "people.cn", "peopleapp.com", "pdnews.cn",
    "baidu.com", "baiducontent.com", "baidubce.com", "bcebos.com", "bdimg.com", "bdstatic.com",
    "bilibili.com", "bilivideo.com", "bilivideo.cn", "hdslb.com",
    "qq.com", "gtimg.com", "gtimg.cn", "qpic.cn", "smtcdns.com",
    "youku.com", "ykimg.com", "iqiyi.com", "qiyi.com", "iqiyipic.com", "qiyipic.com",
    "71edge.com",  # iQiyi's public player dispatches ordinary media to this CDN.
    "sohu.com", "sohu.com.cn", "itc.cn", "cdn.sohucs.com",
    "cnr.cn", "cri.cn", "sina.com.cn", "sinaimg.cn", "weibo.com", "weibocdn.com",
)

# CCTV's public video-info API returns its ordinary HLS URL on this shared CDN.
# Authorize only the observed host, not other tenants or subdomains.
EXACT_HOSTS = frozenset({
    "hls.cntv.cdn20.com",
    "pexels.com", "www.pexels.com", "api.pexels.com", "images.pexels.com", "videos.pexels.com", "static-videos.pexels.com",
    "pixabay.com", "www.pixabay.com", "cdn.pixabay.com",
    "coverr.co", "www.coverr.co", "api.coverr.co", "storage.coverr.co",
    "www.nasa.gov", "images.nasa.gov", "images-api.nasa.gov", "images-assets.nasa.gov",
    "commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org",
})


def allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return host in EXACT_HOSTS or any(host == domain or host.endswith("." + domain) for domain in DOMAINS)


def require(url: str) -> None:
    if not allowed(url):
        raise ValueError("此链接不在已接入的公开素材来源或媒体域名中")
