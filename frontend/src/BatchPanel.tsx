import {useEffect,useState} from 'react';
import {Plus,Upload,X} from 'lucide-react';

type Item={title:string;script:string};
type Row={id:string;project_id:string;title:string;status:string;progress:number;message:string;error:string;kind:string};
async function request(path:string,body?:unknown){const r=await fetch('/api'+path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'请求内容无效');return d}
const names:Record<string,string>={queued:'排队中',running:'制作中',completed:'已完成',needs_review:'待补素材 / 核对',failed:'失败，可重试',cancelled:'已取消',interrupted:'已中断'};

export function BatchPanel({close,openProject,onCreated}:{close:()=>void;openProject:(id:string)=>void;onCreated:()=>void}){
 const [items,setItems]=useState<Item[]>([]),[text,setText]=useState(''),[rows,setRows]=useState<Row[]>([]),[error,setError]=useState(''),[queueError,setQueueError]=useState(''),[busy,setBusy]=useState(false);
 const [aspect,setAspect]=useState('16:9'),[policy,setPolicy]=useState('video_preferred'),[resolution,setResolution]=useState('720p');
 const refresh=()=>request('/queue').then((r:Row[])=>{setRows(r.filter(j=>j.kind==='produce'));setQueueError('')}).catch(()=>setQueueError('工作台暂时连接中断，恢复后会自动更新队列。'));
 useEffect(()=>{refresh();const t=setInterval(refresh,2500);return()=>clearInterval(t)},[]);
 const act=async(fn:()=>Promise<void>)=>{setBusy(true);setError('');try{await fn()}catch(e){setError((e as Error).message)}finally{setBusy(false)}};
 const append=(next:Item[])=>{setItems(old=>[...old,...next].slice(0,20));if(items.length+next.length>20)setError('每次最多 20 篇，超过部分未加入')};
 const importFiles=(files:FileList|null)=>act(async()=>{
  if(!files)return;const next:Item[]=[];
  for(const f of Array.from(files).slice(0,20)){const bytes=await f.arrayBuffer();let script;try{script=new TextDecoder('utf-8',{fatal:true}).decode(bytes)}catch{script=new TextDecoder('gb18030').decode(bytes)};if(script.trim())next.push({title:f.name.replace(/\.(txt|md)$/i,'').slice(0,100),script:script.trim()})}append(next);
 });
 const submit=()=>act(async()=>{await request('/batch',{projects:items.map(i=>({...i,aspect,media_policy:policy})),resolution});setItems([]);onCreated();await refresh()});
 return <div className="modal-backdrop"><section className="modal batch-modal" role="dialog" aria-modal="true" aria-label="批量制作"><div className="modal-heading"><div><h2>批量制作</h2></div><button className="icon-button" aria-label="关闭批量制作" onClick={close}><X/></button></div><div className="modal-content">
  <p>多篇文稿依次完成分镜、配音、配素材和带字幕的视频。需要核对的项目单独保留，后面的文稿继续制作。</p>
  <label className="secondary small"><Upload size={16}/>导入多篇 TXT / MD<input hidden type="file" multiple accept=".txt,.md,text/plain" disabled={busy} onChange={e=>{importFiles(e.target.files);e.target.value=''}}/></label>
  <label>粘贴文稿 · 多篇之间用单独一行 --- 分隔<textarea rows={5} value={text} onChange={e=>setText(e.target.value)} placeholder="每篇文稿生成一个独立项目"/></label>
  <button className="secondary small" disabled={!text.trim()||busy} onClick={()=>{append(text.split(/^\s*---\s*$/m).filter(x=>x.trim()).map(s=>({script:s.trim(),title:s.trim().split('\n')[0].slice(0,36)})));setText('')}}><Plus size={16}/>加入本次列表</button>
  {items.map((item,i)=><div className="batch-item" key={i}><span>{i+1}</span><input aria-label={`第 ${i+1} 篇标题`} value={item.title} maxLength={100} onChange={e=>setItems(items.map((v,j)=>j===i?{...v,title:e.target.value}:v))}/><small>{item.script.length} 字</small><button className="icon-button" aria-label={`移除第 ${i+1} 篇`} onClick={()=>setItems(items.filter((_,j)=>j!==i))}><X size={16}/></button></div>)}
  <div className="field-grid"><label>画面比例<select value={aspect} onChange={e=>setAspect(e.target.value)}><option>16:9</option><option>9:16</option></select></label><label>素材组合<select value={policy} onChange={e=>setPolicy(e.target.value)}><option value="mixed">图片与视频混合（不限制比例）</option><option value="video_preferred">至少 80% 实拍/录屏视频</option><option value="video_only">全部实拍/录屏视频</option></select></label><label>输出<select value={resolution} onChange={e=>setResolution(e.target.value)}><option value="720p">720p · 快速测试</option><option value="1080p">1080p</option></select></label></div>
  <button className="primary" disabled={busy||!items.length||items.some(i=>!i.title.trim()||i.script.length>30000)} onClick={submit}>{busy?'正在提交…':`开始制作 ${items.length} 篇`}</button>
  {error&&<p role="alert" className="candidate-warning">{error}</p>}
  {queueError&&<p role="status" className="candidate-warning">{queueError}</p>}
  <h3>制作队列</h3>{!rows.length&&<p>还没有整片制作任务。</p>}{rows.map(row=><div className="queue-row" key={row.id}><div><button className="text-link" onClick={()=>{close();openProject(row.project_id)}}>{row.title}</button><span>{names[row.status]||row.status}</span></div><progress value={row.progress} max={100}/><p>{row.message}{row.error&&` · ${row.error}`}</p>{['failed','needs_review','cancelled','interrupted'].includes(row.status)&&<button className="secondary small" disabled={busy} onClick={()=>act(async()=>{await request(`/jobs/${row.id}/retry`,{});await refresh()})}>从未完成处继续</button>}{['running','queued'].includes(row.status)&&<button className="text-link" disabled={busy} onClick={()=>act(async()=>{await request(`/jobs/${row.id}/cancel`,{});await refresh()})}>取消</button>}</div>)}
 </div><div className="modal-footer"><small>默认逐篇制作，每篇内部并发检索。使用已保存的模型和配音设置。</small><button className="secondary" onClick={close}>关闭</button></div></section></div>;
}
