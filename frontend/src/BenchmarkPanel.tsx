import {useEffect,useState} from 'react';
import {Copy,ExternalLink,FolderOpen,Loader2,RefreshCw,Sparkles,Trash2,UserPlus,WandSparkles,X} from 'lucide-react';

type Author={id:string;name:string;url:string;fans:string;count:string;scanned:number;error:string};
type Video={vid:string;title:string;author_name:string;plays:number;published_text:string;duration:number;url:string;status:string};
type Draft={name:string;title:string;chars:number;text:string};
type Work={running:boolean;done:number;total:number;current:string;errors:string[];finished:number};
type State={authors:Author[];scanned:number;stale:boolean;videos:Video[];known:number;folders:Record<string,string>;work:Work;pending:string[];review:Draft[]};
async function request<T>(path:string,method='GET',body?:unknown):Promise<T>{const r=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'请求内容无效');return d}
const plays=(n:number)=>n>=10000?(n/10000).toFixed(n>=100000?0:1)+'万':String(n);
const clock=(s:number)=>`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
const when=(t:number)=>t?new Date(t*1000).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}):'还没刷新';

export function BenchmarkPanel({close,onCreated}:{close:()=>void;onCreated:()=>void}){
 const [s,setS]=useState<State|null>(null),[links,setLinks]=useState(''),[days,setDays]=useState('3'),[min,setMin]=useState('10000'),[done,setDone]=useState(false);
 const [picked,setPicked]=useState<string[]>([]),[chosen,setChosen]=useState<string[]>([]),[open,setOpen]=useState(''),[edit,setEdit]=useState('');
 const [aspect,setAspect]=useState('16:9'),[resolution,setResolution]=useState('1080p'),[busy,setBusy]=useState(''),[error,setError]=useState(''),[note,setNote]=useState('');
 const q=`?days=${days}&plays=${min}&done=${done}`;
 const load=()=>request<State>('/benchmark'+q).then(setS).catch(e=>setError(e.message));
 const act=async(label:string,fn:()=>Promise<void>)=>{setBusy(label);setError('');setNote('');try{await fn()}catch(e){setError((e as Error).message)}finally{setBusy('')}};
 const scan=()=>act('scan',async()=>{setS(await request<State>(`/benchmark/scan?days=${days}&plays=${min}`,'POST'))});
 useEffect(()=>{request<State>('/benchmark'+q).then(d=>{setS(d);if(d.stale&&d.authors.length)scan()}).catch(e=>setError(e.message))},[]);
 useEffect(()=>{load()},[days,min,done]);
 useEffect(()=>{if(!s?.work.running)return;const t=setInterval(load,3000);return()=>clearInterval(t)},[s?.work.running]);
 const copy=()=>act('copy',async()=>{const rows=(s?.videos||[]).filter(v=>picked.includes(v.vid)),text=rows.map(v=>v.url).join('\n');
  try{await navigator.clipboard.writeText(text)}catch{const t=document.createElement('textarea');t.value=text;document.body.appendChild(t);t.select();document.execCommand('copy');t.remove()}
  const r=await request<{covers:number}>('/benchmark/mark','POST',{vids:picked,status:'copied'});setPicked([]);await load();setNote(`已复制 ${rows.length} 个链接，封面存了 ${r.covers} 张`)});
 const videos=s?.videos||[],review=s?.review||[],work=s?.work;
 return <div className="modal-backdrop"><section className="modal news-modal bench-modal" role="dialog" aria-modal="true" aria-label="对标采集"><div className="modal-heading"><div><h2>对标采集</h2></div><button className="icon-button" aria-label="关闭对标采集" onClick={close}><X/></button></div><div className="modal-content">
  <h3>对标作者 · {s?.authors.length||0}</h3>
  <div className="bench-add"><textarea aria-label="作者主页" rows={2} placeholder="好看视频作者主页，一行一个" value={links} onChange={e=>setLinks(e.target.value)}/><button className="secondary small" disabled={!!busy||!links.trim()} onClick={()=>act('add',async()=>{setS(await request<State>('/benchmark/authors','POST',{links}));setLinks('');await scan()})}><UserPlus size={15}/>添加</button></div>
  <div className="bench-authors">{(s?.authors||[]).map(a=><div key={a.id} className="bench-author"><span><strong title={a.name||a.id}>{a.name||a.id}</strong>{a.error?<small className="error">{a.error}</small>:<small>{[a.fans&&`粉丝 ${a.fans}`,a.count].filter(Boolean).join(' · ')}</small>}</span><a className="icon-button" aria-label={`打开 ${a.name||a.id} 的主页`} href={a.url} target="_blank" rel="noreferrer noopener"><ExternalLink size={14}/></a><button className="icon-button" aria-label={`删除 ${a.name||a.id}`} disabled={!!busy} onClick={()=>act('remove',async()=>{setS(await request<State>('/benchmark/authors/'+a.id,'DELETE'));await load()})}><Trash2 size={14}/></button></div>)}</div>

  <h3>爆款 · {videos.length}{picked.length>0&&<span className="news-picked">已选 {picked.length}</span>}
   {videos.length>0&&<span className="news-pick-all"><button className="text-link" onClick={()=>setPicked(videos.map(v=>v.vid))}>全选</button>{picked.length>0&&<button className="text-link" onClick={()=>setPicked([])}>全不选</button>}</span>}</h3>
  <div className="bench-bar"><label>发布<select value={days} onChange={e=>setDays(e.target.value)}><option value="1">1 天内</option><option value="3">3 天内</option><option value="7">7 天内</option><option value="30">30 天内</option><option value="0">不限</option></select></label>
   <label>播放<select value={min} onChange={e=>setMin(e.target.value)}>{[['1000','1000 以上'],['5000','5000 以上'],['10000','1 万以上'],['50000','5 万以上'],['100000','10 万以上']].map(([v,t])=><option key={v} value={v}>{t}</option>)}</select></label>
   <label><input type="checkbox" checked={done} onChange={e=>setDone(e.target.checked)}/>含已复制</label>
   <button className="secondary small" disabled={!!busy||!s?.authors.length} onClick={scan}>{busy==='scan'?<Loader2 className="spin" size={15}/>:<RefreshCw size={15}/>}刷新</button><small>{when(s?.scanned||0)}</small>
   <button className="secondary small" disabled={!!busy||!picked.length} onClick={copy}><Copy size={15}/>复制链接{picked.length?`（${picked.length}）`:''}</button></div>
  <div className="news-list">{videos.map(v=><div key={v.vid} className={'news-item '+(picked.includes(v.vid)?'picked':'')}>
   <label className="checkbox"><input type="checkbox" checked={picked.includes(v.vid)} onChange={()=>setPicked(p=>p.includes(v.vid)?p.filter(x=>x!==v.vid):[...p,v.vid])}/><span><strong>{v.title}</strong><small>{[v.author_name,`${plays(v.plays)} 播放`,v.published_text,v.duration?clock(v.duration):'',v.status==='copied'?'已复制':''].filter(Boolean).join(' · ')}</small></span></label>
   <div className="news-actions"><a className="text-link" href={v.url} target="_blank" rel="noreferrer noopener"><ExternalLink size={12}/>打开</a></div></div>)}
   {!videos.length&&<p className="news-empty">{s?.authors.length?'没有符合条件的视频':'先添加对标作者'}</p>}</div>

  <h3>转写洗稿</h3>
  <div className="bench-bar"><button className="secondary small" onClick={()=>act('open1',async()=>{await request('/benchmark/open/1-放视频','POST')})}><FolderOpen size={15}/>1-放视频</button><button className="secondary small" onClick={()=>act('open2',async()=>{await request('/benchmark/open/2-待审','POST')})}><FolderOpen size={15}/>2-待审</button><button className="secondary small" onClick={()=>act('open3',async()=>{await request('/benchmark/open/封面','POST')})}><FolderOpen size={15}/>封面</button>
   <button className="secondary small" disabled={!!busy||!!work?.running||!s?.pending.length} onClick={()=>act('process',async()=>{await request('/benchmark/process','POST');await load()})}>{work?.running?<Loader2 className="spin" size={15}/>:<WandSparkles size={15}/>}{work?.running?`转写洗稿 ${work.done}/${work.total}`:`转写洗稿${s?.pending.length?`（${s.pending.length} 个）`:''}`}</button></div>
  {work?.running&&work.current&&<small className="news-hint">{work.current}</small>}
  {!!work?.errors.length&&<div className="candidate-warning">{work.errors.map((e,i)=><p key={i}>{e}</p>)}</div>}

  <h3>待审 · {review.length}{chosen.length>0&&<span className="news-picked">已选 {chosen.length}</span>}
   {review.length>0&&<span className="news-pick-all"><button className="text-link" onClick={()=>setChosen(review.map(r=>r.name))}>全选</button>{chosen.length>0&&<button className="text-link" onClick={()=>setChosen([])}>全不选</button>}</span>}</h3>
  <div className="news-list">{review.map(r=><div key={r.name} className={'news-item '+(chosen.includes(r.name)?'picked ':'')+(open===r.name?'open':'')}>
   <label className="checkbox"><input type="checkbox" checked={chosen.includes(r.name)} onChange={()=>setChosen(c=>c.includes(r.name)?c.filter(x=>x!==r.name):[...c,r.name])}/><span><strong>{r.title}</strong><small>{r.chars} 字</small></span></label>
   <div className="news-actions"><button className="text-link" onClick={()=>{setOpen(open===r.name?'':r.name);setEdit(r.text)}}>{open===r.name?'收起':'看稿'}</button><button className="icon-button" aria-label={`删除 ${r.title}`} disabled={!!busy} onClick={()=>act('drop',async()=>{await request('/benchmark/review/delete','POST',{name:r.name});setChosen(c=>c.filter(x=>x!==r.name));await load()})}><Trash2 size={14}/></button></div>
   {open===r.name&&<div className="news-paste"><textarea aria-label={`修改 ${r.title}`} rows={10} value={edit} onChange={e=>setEdit(e.target.value)}/><button className="secondary small" disabled={!!busy||edit===r.text} onClick={()=>act('save',async()=>{await request('/benchmark/review','PUT',{name:r.name,text:edit});await load();setNote('已保存')})}>保存</button></div>}
  </div>)}{!review.length&&<p className="news-empty">洗好的稿会出现在这里</p>}</div>
  <div className="bench-bar"><label>画面比例<select value={aspect} onChange={e=>setAspect(e.target.value)}><option value="16:9">16:9 横屏</option><option value="9:16">9:16 竖屏</option></select></label><label>输出<select value={resolution} onChange={e=>setResolution(e.target.value)}><option value="1080p">1080p</option><option value="720p">720p · 更快</option></select></label></div>
  <button className="primary" disabled={!!busy||!chosen.length} onClick={()=>act('produce',async()=>{const r=await request<{projects:unknown[]}>('/benchmark/produce','POST',{names:chosen,aspect,resolution,burn_subtitles:false});setChosen([]);await load();onCreated();setNote(`已送进批量 ${r.projects.length} 篇`)})}>{busy==='produce'?<Loader2 className="spin" size={16}/>:<Sparkles size={16}/>}{chosen.length?`送进批量 ${chosen.length} 篇`:'勾选待审的稿子'}</button>
  {error&&<p role="alert" className="candidate-warning">{error}</p>}
  {note&&<p role="status" className="news-note">{note}</p>}
 </div><div className="modal-footer"><button className="secondary" onClick={close}>关闭</button></div></section></div>;
}
