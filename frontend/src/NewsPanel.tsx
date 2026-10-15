import {useEffect,useState} from 'react';
import {ExternalLink,FileText,Link2,Loader2,Newspaper,RefreshCw,Sparkles,Trash2,X} from 'lucide-react';

type Item={id:string;title:string;text:string;url:string;source:string;published:string;origin:string;added:number;chars:number;projects:string[];koubo?:string;koubo_title?:string};
type Crawler={ready:boolean;builtin?:boolean;message:string;dir:string;latest?:{name:string;count:number}|null};
type Row={id:string;project_id:string;title:string;status:string;progress:number;message:string;error:string;kind:string};
async function request<T>(path:string,method='GET',body?:unknown):Promise<T>{const r=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'请求内容无效');return d}
const names:Record<string,string>={queued:'排队中',running:'制作中',completed:'已完成',needs_review:'待补素材 / 核对',failed:'失败，可重试',cancelled:'已取消',interrupted:'已中断'};
const kinds:Record<string,string>={render:'导出',match:'配素材',voice:'配音',voice_full:'整段配音',plan:'分镜',search:'搜索素材',verify:'核验',select:'选片'};
const origins:Record<string,string>={crawler:'爬虫',link:'链接',text:'手动粘贴'};

export function NewsPanel({close,openProject,onCreated}:{close:()=>void;openProject:(id:string)=>void;onCreated:()=>void}){
 const [items,setItems]=useState<Item[]>([]),[crawler,setCrawler]=useState<Crawler|null>(null),[picked,setPicked]=useState<string[]>([]),[open,setOpen]=useState('');
 const [url,setUrl]=useState(''),[title,setTitle]=useState(''),[text,setText]=useState(''),[busy,setBusy]=useState(''),[error,setError]=useState(''),[note,setNote]=useState('');
 const [withScripts,setWithScripts]=useState(false),[aspect,setAspect]=useState('16:9'),[minutes,setMinutes]=useState('4'),[resolution,setResolution]=useState('720p'),[rows,setRows]=useState<Row[]>([]);
 const load=()=>request<{items:Item[];crawler:Crawler}>('/news').then(d=>{setItems(d.items);setCrawler(d.crawler)}).catch(e=>setError(e.message));
 // Each project's latest job of any kind: a run finished later (e.g. an export) supersedes an older failed one.
 const queue=()=>request<Row[]>('/queue').then(r=>{const seen=new Set<string>();setRows(r.filter(j=>!seen.has(j.project_id)&&!!seen.add(j.project_id)))}).catch(()=>{});
 useEffect(()=>{load();queue();const t=setInterval(queue,2500);return()=>clearInterval(t)},[]);
 const [bottom,setBottom]=useState(false);
 const act=async(label:string,fn:()=>Promise<void>)=>{setBusy(label);setError('');setNote('');setBottom(['produce','open','retry'].includes(label));try{await fn()}catch(e){setError((e as Error).message)}finally{setBusy('')}};
 const newsProjects=new Set(items.flatMap(i=>i.projects||[]));
 const produce=(auto:boolean,ids:string[])=>act(auto?'produce':'open',async()=>{
  const r=await request<{projects:{project_id:string}[]}>('/news/produce','POST',{ids,aspect,minutes:minutes?Number(minutes):null,resolution,auto});
  setPicked([]);await load();onCreated();
  if(!auto&&r.projects[0]){close();openProject(r.projects[0].project_id)}else{setNote(`已加入制作队列 ${r.projects.length} 篇：先 AI 洗稿并核对事实，再配素材、配音、出片。`);queue()}
 });
 const toggle=(id:string)=>setPicked(p=>p.includes(id)?p.filter(x=>x!==id):p.length>=200?p:[...p,id]);
 const fresh=items.filter(i=>!i.projects?.length).map(i=>i.id);
 return <div className="modal-backdrop"><section className="modal news-modal" role="dialog" aria-modal="true" aria-label="新闻选题"><div className="modal-heading"><div><h2>新闻选题与洗稿</h2></div><button className="icon-button" aria-label="关闭新闻选题" onClick={close}><X/></button></div><div className="modal-content">
  <p>从新闻爬虫、新闻链接或粘贴的原文开始。AI 先改写成口播稿（保留人名、数字和事实，对照原文核对），再自动配素材、配音并导出视频。</p>
  <div className="news-sources">
   <button className="secondary small" disabled={!!busy||!crawler?.ready} title={crawler?.ready?'运行新闻爬虫，抓取最新新闻（约 1-3 分钟）':crawler?.message} onClick={()=>act('crawl',async()=>{const r=await request<{fetched:number;new:number;scripts:number;items:Item[]}>('/news/crawl','POST',{koubo:withScripts});setItems(r.items);await load();setNote(`爬虫抓到 ${r.fetched} 条，新增 ${r.new} 条${r.scripts?`，其中 ${r.scripts} 条已有爬虫口播稿`:''}`)})}>{busy==='crawl'?<Loader2 className="spin" size={15}/>:<RefreshCw size={15}/>}{busy==='crawl'?'正在抓取，约 1-3 分钟…':'抓取最新新闻'}</button>
   <div className="search-row"><input aria-label="新闻链接" placeholder="粘贴新闻链接，例如 https://www.chinanews.com.cn/…" value={url} onChange={e=>setUrl(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&url.trim())(e.currentTarget.nextSibling as HTMLButtonElement)?.click()}}/><button className="secondary small" disabled={!!busy||!url.trim()} onClick={()=>act('link',async()=>{const item=await request<Item>('/news/link','POST',{url:url.trim()});setUrl('');await load();setOpen(item.id);setNote(`已读取：${item.title}（${item.chars} 字）`)})}>{busy==='link'?<Loader2 className="spin" size={15}/>:<Link2 size={15}/>}读取</button></div>
  </div>
  {crawler&&!crawler.ready&&<small className="news-hint">{crawler.message}</small>}
  {crawler?.builtin&&<div className="news-crawler-row"><label className="checkbox"><input type="checkbox" checked={withScripts} disabled={!!busy} onChange={e=>setWithScripts(e.target.checked)}/>抓取时让爬虫用通义千问写口播稿（爬虫自带提示词，每次最多 5 篇、每篇约 0.02 元；不勾选则在出片时由本工具洗稿）</label>
   <button className="text-link" disabled={!!busy||!crawler.latest} title={crawler.latest?`导入 ${crawler.latest.name}（${crawler.latest.count} 篇）`:'爬虫还没有抓取过'} onClick={()=>act('import',async()=>{const r=await request<{fetched:number;new:number;scripts:number;items:Item[]}>('/news/import-latest','POST');setItems(r.items);setNote(`导入 ${r.fetched} 条，新增 ${r.new} 条${r.scripts?`，其中 ${r.scripts} 条带爬虫口播稿`:''}`)})}>导入上次抓取结果{crawler.latest?`（${crawler.latest.count} 篇）`:''}</button></div>}
  <details className="news-paste"><summary>手动粘贴新闻原文</summary><input aria-label="新闻标题" placeholder="标题（可选）" value={title} maxLength={200} onChange={e=>setTitle(e.target.value)}/><textarea aria-label="新闻原文" rows={5} placeholder="粘贴新闻正文" value={text} onChange={e=>setText(e.target.value)}/><button className="secondary small" disabled={!!busy||text.trim().length<30} onClick={()=>act('text',async()=>{const item=await request<Item>('/news/text','POST',{title,text});setTitle('');setText('');await load();setOpen(item.id)})}><FileText size={15}/>加入选题</button></details>
  {!bottom&&error&&<p role="alert" className="candidate-warning">{error}</p>}
  {!bottom&&note&&<p role="status" className="news-note">{note}</p>}
  <h3><Newspaper size={16}/>选题 · {items.length} 条{picked.length>0&&<span className="news-picked">已选 {picked.length}</span>}
   {items.length>0&&<span className="news-pick-all"><button className="text-link" onClick={()=>setPicked(items.map(i=>i.id).slice(0,200))}>全选</button>{fresh.length>0&&fresh.length<items.length&&<button className="text-link" onClick={()=>setPicked(fresh.slice(0,200))}>只选没做过的（{fresh.length}）</button>}{picked.length>0&&<button className="text-link" onClick={()=>setPicked([])}>全不选</button>}</span>}</h3>
  {!items.length&&<p className="news-empty">还没有新闻。抓取最新新闻、粘贴链接或原文后会显示在这里。</p>}
  <div className="news-list">{items.map(item=><div key={item.id} className={'news-item '+(picked.includes(item.id)?'picked':'')}>
   <label className="checkbox"><input type="checkbox" checked={picked.includes(item.id)} onChange={()=>toggle(item.id)}/><span><strong>{item.title}</strong><small>{[item.source,item.published&&item.published.slice(0,16),`${item.chars} 字`,origins[item.origin]||item.origin,item.koubo?'已有爬虫口播稿':'',item.projects?.length?`已做 ${item.projects.length} 次`:''].filter(Boolean).join(' · ')}</small></span></label>
   <div className="news-actions"><button className="text-link" onClick={()=>setOpen(open===item.id?'':item.id)}>{open===item.id?'收起':'原文'}</button>{item.url&&<a className="text-link" href={item.url} target="_blank" rel="noreferrer noopener"><ExternalLink size={12}/>来源</a>}<button className="text-link" disabled={!!busy} title="建一个项目，原文放进文稿；在文稿页点 AI 洗稿后可先看稿再生成" onClick={()=>produce(false,[item.id])}>先看稿</button><button className="icon-button" aria-label={`删除 ${item.title}`} disabled={!!busy} onClick={()=>act('delete',async()=>{await request('/news/'+item.id,'DELETE');setPicked(p=>p.filter(x=>x!==item.id));await load()})}><Trash2 size={14}/></button></div>
   {open===item.id&&<pre className="news-text">{item.text}</pre>}
  </div>)}</div>
  <div className="field-grid three"><label>画面比例<select value={aspect} onChange={e=>setAspect(e.target.value)}><option value="16:9">16:9 横屏</option><option value="9:16">9:16 竖屏</option></select></label><label>口播最长<select value={minutes} onChange={e=>setMinutes(e.target.value)}><option value="1">1 分钟</option><option value="2">2 分钟</option><option value="3">3 分钟</option><option value="4">4 分钟</option><option value="6">6 分钟</option><option value="">按原文长度</option></select></label><label>输出<select value={resolution} onChange={e=>setResolution(e.target.value)}><option value="720p">720p · 更快</option><option value="1080p">1080p</option></select></label></div>
  <button className="primary" disabled={!!busy||!picked.length} onClick={()=>produce(true,picked)}>{busy==='produce'?<Loader2 className="spin" size={16}/>:<Sparkles size={16}/>}{picked.length?`洗稿并自动出片 ${picked.length} 篇`:'勾选新闻后洗稿并自动出片'}</button>
  {bottom&&error&&<p role="alert" className="candidate-warning">{error}</p>}
  {bottom&&note&&<p role="status" className="news-note">{note}</p>}
  {rows.some(r=>newsProjects.has(r.project_id))&&<><h3>新闻制作队列{rows.some(r=>newsProjects.has(r.project_id)&&['running','queued'].includes(r.status))&&<button className="secondary small danger" disabled={!!busy} onClick={()=>{if(confirm('取消全部排队和制作中的视频？已经做好的不受影响。'))act('cancel',async()=>{await request('/queue/cancel','POST',{});await queue()})}}>全部取消</button>}</h3>{rows.filter(r=>newsProjects.has(r.project_id)).slice(0,200).map(row=><div className="queue-row" key={row.id}><div><button className="text-link" onClick={()=>{close();openProject(row.project_id)}}>{row.title}</button><span>{row.kind!=='produce'?(kinds[row.kind]||row.kind)+' · ':''}{names[row.status]||row.status}</span></div><progress value={row.progress} max={100}/><p>{row.message}{row.error&&` · ${row.error}`}</p>{row.kind==='produce'&&['failed','needs_review','cancelled','interrupted'].includes(row.status)&&<button className="secondary small" disabled={!!busy} onClick={()=>act('retry',async()=>{await request(`/jobs/${row.id}/retry`,'POST',{});await queue()})}>从未完成处继续</button>}</div>)}</>}
 </div><div className="modal-footer"><small>每篇洗稿约 0.01-0.02 元（两次模型请求），计入该项目费用。</small><button className="secondary" onClick={close}>关闭</button></div></section></div>;
}

export function NewsCrawlerSettings({values,change}:{values:Record<string,any>;change:(key:string,value:string)=>void}){
 return <section><h3><Newspaper size={16}/>新闻爬虫</h3>
  <p>填写“全平台新闻爬虫”的文件夹即可自动识别：抓取今日头条、腾讯新闻、搜狐网等平台的新闻，也能导入在爬虫窗口里抓好的批次和它生成的口播稿。爬虫需要写口播稿时，临时使用上方的通义千问接口与密钥（不写入爬虫的配置文件）。其他爬虫可填写运行命令和结果文件（JSON / JSONL / CSV / SQLite）。</p>
  <label>爬虫目录<input value={values.news_crawler_dir||''} placeholder="例如 /Users/me/news-crawler 或 D:\新闻爬虫" onChange={e=>change('news_crawler_dir',e.target.value)}/></label>
  <div className="field-grid"><label>每个平台抓取条数<select value={values.news_crawler_limit||'5'} onChange={e=>change('news_crawler_limit',e.target.value)}>{['3','5','10','20','30'].map(n=><option key={n} value={n}>{n} 条</option>)}</select></label><label>平台（留空用爬虫默认）<input value={values.news_crawler_platforms||''} placeholder="toutiao,tencent,sohu" onChange={e=>change('news_crawler_platforms',e.target.value)}/></label></div>
  <details className="news-paste"><summary>其他爬虫：运行命令与结果文件</summary><div className="field-grid"><label>运行命令<input value={values.news_crawler_command||''} placeholder="例如 python crawl.py" onChange={e=>change('news_crawler_command',e.target.value)}/></label><label>结果文件（相对爬虫目录）<input value={values.news_crawler_output||''} placeholder="例如 output/news.json" onChange={e=>change('news_crawler_output',e.target.value)}/></label></div></details>
 </section>;
}
