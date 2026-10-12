import base64
import json

import pytest

from jinggao import domestic, paths, platform_videos, portal_videos, video_sources


def iqiyi_card(**extra):
    return {"title":"马龙乒乓球采访","pageUrl":"https://www.iqiyi.com/v_u4dmtp3hxk.html",
            "img":"https://pic1.iqiyipic.com/one.jpg","duration":25000,
            "qipuId":3003008124784100,"playUrl":"qips://tvid=3003008124784100;ischarge=false;", **extra}


def test_iqiyi_cards_are_individual_public_videos_not_parent_biographies():
    body = {"code":0,"data":{"templates":[{"albumInfo":{"title":"同名演员",
        "brief":{"value":"演员个人介绍不是运动员资料"},"groupedVideos":[{"videos":[iqiyi_card()]}]}},
        {"albumInfo":iqiyi_card(pageUrl="https://sports.iqiyi.com/resource/pcw/play/mvnqmnp7hk?playertype=immersiveShort",
            qipuId=2309134666860100,
            uploader={"value":"体育编辑"},releaseTime={"value":"2026-10-01"})},
        {"albumInfo":iqiyi_card(firstVideoIsVip=True)},
        {"albumInfo":iqiyi_card(pageUrl="https://www.iqiyi.com/a_abc.html")}]}}
    rows = portal_videos.parse_iqiyi_search(body)
    assert len(rows)==2 and rows[0]['duration']==25
    assert not rows[0]['description'] and '演员个人' not in json.dumps(rows,ensure_ascii=False)
    assert rows[1]['publisher']=='体育编辑' and rows[1]['published_at']=='2026-10-01'
    assert all(r['download_status']=='unchecked' for r in rows)


def test_iqiyi_foreign_paid_and_album_cards_are_excluded():
    cards = [iqiyi_card(img='https://overseas.test/cover.jpg'),iqiyi_card(payMarkUrl='vip.png'),
             iqiyi_card(playUrl='qips://tvid=123;ischarge=true;'),iqiyi_card(duration=0),
             iqiyi_card(qipuId=2309134666860100)]
    assert portal_videos.parse_iqiyi_search({'code':0,'data':cards})==[]


def test_sohu_catalogue_excludes_collections_and_uses_own_card_title():
    path = base64.urlsafe_b64encode(b'us/123/456.shtml').decode()
    collection = base64.urlsafe_b64encode(b'pl/123.shtml').decode()
    html = f'<a href="//tv.sohu.com/v/{path}.html" title="厨师制作面条"><img src="//cover.cdn.sohucs.com/cover.jpg"></a>'
    html += f'<a href="//tv.sohu.com/v/{collection}.html" title="整个美食频道"><img src="//cover.cdn.sohucs.com/a.jpg"></a>'
    rows = portal_videos.parse_cards(html,'sohu')
    assert len(rows)==1 and rows[0]['title']=='厨师制作面条' and rows[0]['video_id']=='456'
    assert domestic.allowed(rows[0]['thumbnail_url'])


def test_youku_challenge_stops_provider_wide_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'DATA',tmp_path)
    monkeypatch.setattr(portal_videos,'_cooldown',{})
    calls = []
    monkeypatch.setattr(video_sources,'_fetch',lambda *a,**kw:(calls.append(a) or b'_____tmd_____',{}))
    for term in ('采访','城市','美食'):
        with pytest.raises(ValueError,match='验证'):
            portal_videos.youku_search(term)
    assert len(calls)==1


def test_sohu_multivideo_is_not_mislabelled_as_one_complete_video(monkeypatch):
    path = base64.urlsafe_b64encode(b'us/123/456.shtml').decode()
    source = f'https://tv.sohu.com/v/{path}.html'
    monkeypatch.setattr(video_sources,'_ytdlp_command',lambda:['yt-dlp'])
    monkeypatch.setattr(video_sources,'_run_ytdlp_json',lambda *a,**kw:{'id':'456','_type':'multi_video','entries':[{}]})
    with pytest.raises(ValueError,match='单个视频'):
        platform_videos.inspect_platform({'source_url':source},lambda:None)


def test_youku_requires_unencrypted_domestic_streams(monkeypatch):
    source='https://v.youku.com/v_show/id_abcd.html'
    monkeypatch.setattr(video_sources,'_ytdlp_command',lambda:['yt-dlp'])
    body={'id':'abcd','webpage_url':source,'formats':[{'url':'https://example.com/a.mp4','ext':'mp4','protocol':'https','width':1280,'height':720}]}
    monkeypatch.setattr(video_sources,'_run_ytdlp_json',lambda *a,**kw:body)
    with pytest.raises(ValueError,match='国内视频流'):
        platform_videos.inspect_platform({'source_url':source},lambda:None)


def test_sohu_original_dimensions_do_not_hide_a_better_actual_rendition(monkeypatch):
    path=base64.urlsafe_b64encode(b'us/123/456.shtml').decode()
    source=f'https://tv.sohu.com/v/{path}.html'
    monkeypatch.setattr(video_sources,'_ytdlp_command',lambda:['yt-dlp'])
    body={'id':'456','title':'公开烹饪视频','duration':30,'formats':[
        {'format_id':f,'url':f'https://tv.sohu.com/{f}.mp4','ext':'mp4','protocol':'https',
         'width':1920,'height':1080} for f in ['nor','high','super']]}
    monkeypatch.setattr(video_sources,'_run_ytdlp_json',lambda *a,**kw:body)
    def probe(s,*args):
        w,h=(960,540) if 'nor.mp4' in s['url'] else (1280,720) if 'high.mp4' in s['url'] else (1920,1080)
        return {'width':w,'height':h,'duration':30}
    monkeypatch.setattr(video_sources,'probe_stream',probe)
    row=platform_videos.inspect_platform({'source_url':source},lambda:None)
    assert sorted(s['height'] for s in row['media_streams'])==[540,720,1080]
    row['prefer_efficient_stream']=True
    assert video_sources._choose_stream(row,False)['url'].endswith('high.mp4')


def iqiyi_playback_response(ident="3003008124784100", *, info_extra=None, data_extra=None, stream_extra=None):
    stream = {"_selected":True,"drmType":1,"ff":"ts","code":2,"isPreview":0,"ptt":0,
        "scrsz":"1280x720","ml":"https://meta-cdn.video.iqiyi.com/content.m3u8","duration":25,
        **(stream_extra or {})}
    dash = {"code":"A00000","data":{"tvid":int(ident),"st":101,
        "content":{"bossStatus":0,"pm":2,"vipTypes":[0]},"program":{"video":[stream]},**(data_extra or {})}}
    return {"tvid":int(ident),"tvId":int(ident),"videoInfo":{"tvId":int(ident),
        "title":"马龙乒乓球采访","payMark":0,"effective":True,"isLock":False,
        "pagePublishStatus":"PAGE_ONLINE","downloadAllowed":True,**(info_extra or {})},
        "ev":"".join(chr(ord(c)^90) for c in json.dumps(dash,ensure_ascii=False))}


@pytest.mark.parametrize(("url","ident"),[
    ("https://www.iqiyi.com/v_u4dmtp3hxk.html","3003008124784100"),
    ("https://sports.iqiyi.com/resource/pcw/play/mvnqmnp7hk?playertype=immersiveShort","2309134666860100"),
])
def test_iqiyi_new_public_player_routes_match_video_and_manifest(monkeypatch,url,ident):
    calls=[]
    def metadata(endpoint,**kwargs):
        calls.append(endpoint)
        assert "tvid="+ident in endpoint and kwargs['headers']=={'Referer':url}
        assert "disableDRM" not in endpoint and "auth" not in endpoint
        return iqiyi_playback_response(ident)
    monkeypatch.setattr(video_sources,'_json',metadata)
    monkeypatch.setattr(video_sources,'_hls',lambda *a:{'variants':[],'duration':25,
        'segments':[{'url':'https://v-test.71edge.com/video.ts','duration':25,'start':0}]})
    row=platform_videos.inspect_platform({'source_url':url,'video_id':ident},lambda:None)
    assert len(calls)==1 and row['media_extractor']=='iqiyi-public'
    assert row['source_evidence']['video_id']==ident and row['source_verified']
    assert row['media_streams'][0]['height']==720 and row['duration']==25


@pytest.mark.parametrize("changes",[
    {'info_extra':{'tvId':123456}}, {'data_extra':{'tvid':123456}},
    {'info_extra':{'payMark':1}}, {'info_extra':{'isLock':True}},
    {'info_extra':{'downloadAllowed':False}}, {'info_extra':{'pagePublishStatus':'OFFLINE'}},
    {'data_extra':{'content':{'bossStatus':1,'pm':0}}},
    {'data_extra':{'content':{'bossStatus':0,'vipTypes':[4]}}},
    {'data_extra':{'content':{'bossStatus':0,'vipTypes':4}}},
    {'data_extra':{'boss':{'code':'A00000','data':{'prv':1}}}},
    {'data_extra':{'st':703}},
])
def test_iqiyi_rejects_identity_permission_preview_and_region_limits(changes):
    with pytest.raises(ValueError):
        portal_videos.parse_iqiyi_playback(iqiyi_playback_response(**changes),'3003008124784100')


@pytest.mark.parametrize('changes',[
    {'drmType':7},{'drmType':8},{'drmType':9},{'drmType':77},
    {'isPreview':1},{'ptt':1},{'has_drm':True},{'ff':'dash'},
    {'ml':'https://foreign.example/content.m3u8'}, {'scrsz':'720x99999'},
])
def test_iqiyi_only_accepts_selected_clear_domestic_rendition(changes):
    with pytest.raises(ValueError,match='国内视频流'):
        portal_videos.parse_iqiyi_playback(iqiyi_playback_response(stream_extra=changes),'3003008124784100')


def test_iqiyi_candidate_id_mismatch_fails_before_network(monkeypatch):
    monkeypatch.setattr(video_sources,'_json',lambda *a,**kw:pytest.fail('must not request another video'))
    with pytest.raises(ValueError,match='编号不一致'):
        platform_videos.inspect_platform({'source_url':'https://www.iqiyi.com/v_u4dmtp3hxk.html',
            'video_id':'2309134666860100'},lambda:None)


@pytest.mark.parametrize('playlist',[
    '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="https://meta-cdn.video.iqiyi.com/key"\n#EXTINF:25,\nhttps://v-test.71edge.com/video.ts\n#EXT-X-ENDLIST',
    '#EXTM3U\n#EXTINF:25,\nhttps://foreign.example/video.ts\n#EXT-X-ENDLIST',
    '#EXTM3U\n#EXTINF:6,\nhttps://v-test.71edge.com/video.ts\n#EXT-X-ENDLIST',
])
def test_iqiyi_manifest_cannot_introduce_encryption_foreign_media_or_truncated_preview(monkeypatch,playlist):
    monkeypatch.setattr(video_sources,'_json',lambda *a,**kw:iqiyi_playback_response())
    monkeypatch.setattr(video_sources,'_fetch',lambda url,**kw:(playlist.encode(),{'url':url}))
    with pytest.raises(ValueError):
        platform_videos.inspect_platform({'source_url':'https://www.iqiyi.com/v_u4dmtp3hxk.html'},lambda:None)


def test_iqiyi_cdn_suffix_is_exact():
    assert domestic.allowed('https://v-6fce1712.71edge.com/video.ts')
    assert not domestic.allowed('https://71edge.com.attacker.example/video.ts')
    assert not domestic.allowed('https://not71edge.com/video.ts')


def iqiyi_dispatch_fixture():
    ident='3003008124784100'
    source='https://www.iqiyi.com/v_u4dmtp3hxk.html'
    path='/videos/vts/20260717/38/1a/content.ts'
    query='start=0&end=600000&sd=0&qd_tvid='+ident
    segment={'url':'https://data.video.iqiyi.com'+path+'?'+query,'duration':2,'start':0}
    stream={'iqiyi_video_id':ident,'iqiyi_dispatch_files':{'content.ts':'qd_src=public&qd_tvid='+ident}}
    target='https://v-test.71edge.com'+path+'?'+query
    return source,segment,stream,{'uri':path,'d':[{'URL':target}]}


@pytest.mark.parametrize('common',[False,True])
def test_iqiyi_public_dispatch_preserves_supplied_identity_and_byte_range(monkeypatch,common):
    source,segment,stream,body=iqiyi_dispatch_fixture()
    if common:
        stream['iqiyi_dispatch_common']=stream.pop('iqiyi_dispatch_files')['content.ts']
    def dispatch(url,**kw):
        assert url.startswith(segment['url']+'&qd_src=public&qd_tvid=')
        assert '&cross-domain=1&mss=3&stauto=0' in url
        assert kw['headers']=={'Referer':source}
        return body
    monkeypatch.setattr(video_sources,'_json',dispatch)
    assert portal_videos.resolve_iqiyi_segment(segment,stream,source,lambda:None)==body['d'][0]['URL']


@pytest.mark.parametrize('replacement',[
    ('v-test.71edge.com','outside.example'), ('content.ts','advertisement.ts'),
    ('qd_tvid=3003008124784100','qd_tvid=2309134666860100'),
    ('start=0','start=1000'), ('end=600000','end=600000000'), ('sd=0','sd=8000'),
])
def test_iqiyi_dispatch_rejects_wrong_video_cdn_and_byte_range(monkeypatch,replacement):
    source,segment,stream,body=iqiyi_dispatch_fixture()
    body['d'][0]['URL']=body['d'][0]['URL'].replace(*replacement)
    monkeypatch.setattr(video_sources,'_json',lambda *a,**kw:body)
    with pytest.raises(ValueError,match='范围不一致'):
        portal_videos.resolve_iqiyi_segment(segment,stream,source,lambda:None)


def test_iqiyi_dispatch_identity_mismatch_fails_before_network(monkeypatch):
    source,segment,stream,body=iqiyi_dispatch_fixture()
    stream['iqiyi_dispatch_files']['content.ts']='qd_tvid=2309134666860100'
    monkeypatch.setattr(video_sources,'_json',lambda *a,**kw:pytest.fail('must not dispatch another video'))
    with pytest.raises(ValueError,match='编号与原视频不一致'):
        portal_videos.resolve_iqiyi_segment(segment,stream,source,lambda:None)


def test_iqiyi_inspection_cache_keeps_public_dispatch_marker(tmp_path,monkeypatch):
    from jinggao import source_index
    monkeypatch.setattr(paths,'DATA',tmp_path)
    monkeypatch.setattr(video_sources,'_inspection_cache',{})
    monkeypatch.setattr(source_index,'record_check',lambda candidate:None)
    calls=[]
    monkeypatch.setattr(video_sources,'_json',lambda *a,**kw:(calls.append(a) or iqiyi_playback_response()))
    monkeypatch.setattr(video_sources,'_hls',lambda *a:{'variants':[],'duration':25,
        'segments':[{'url':'https://data.video.iqiyi.com/video.ts','duration':25,'start':0}]})
    candidate={'provider':'iqiyi','source_url':'https://www.iqiyi.com/v_u4dmtp3hxk.html',
        'video_id':'3003008124784100'}
    initial=video_sources.inspect_video_candidate(dict(candidate))
    cached=video_sources.inspect_video_candidate(dict(candidate))
    assert len(calls)==1
    assert initial['media_extractor']==cached['media_extractor']=='iqiyi-public'
    assert cached['media_streams'][0]['iqiyi_video_id']==candidate['video_id']
