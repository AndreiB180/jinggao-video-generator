import {useEffect,useState} from 'react';
import {AudioLines,CheckCircle2,FileText,FolderOpen,Plus,RotateCcw,Square,Trash2,Upload,X} from 'lucide-react';
import {useVoices} from './VoicePicker';
import {FolderBar,VideoFileActions,fileName} from './FileActions';

type Item={title:string;script:string};
const AUDIO=/\.(mp3|wav|m4a|aac|flac|ogg|wma|mp4|mov|mkv|webm|avi|flv)$/i;
const megabytes=(n:number)=>n<1048576?`${Math.max(1,Math.round(n/1024))} KB`:`${(n/1048576).toFixed(1)} MB`;
type Row={id:string;project_id:string;title:string;status:string;progress:number;message:string;error:string;kind:string;batch?:string|null;created:number;updated:number;video?:{path:string;saved_to:string;duration:number}|null};
type Costs={model:{daily_cny:number;daily_limit_cny:number}};
async function request(path:string,body?:unknown,method?:string){const r=await fetch('/api'+path,{method:method||(body===undefined?'GET':'POST'),headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'请求内容无效');return d}
const names:Record<string,string>={queued:'排队中',running:'制作中',completed:'已完成',needs_review:'待补素材 / 核对',failed:'失败，可重试',cancelled:'已取消',interrupted:'已中断'};
const MAX=200;
const order:Record<string,number>={running:0,queued:1,failed:2,interrupted:2,needs_review:3,cancelled:4,completed:5};
/** Rough model money for one script (from real runs: ~¥0.012 for 420 字, ~¥0.075 for 3,900 字). */
const costOf=(chars:number)=>.005+chars*.000018;
/** Rough minutes for one video made alone (from real runs: ~4 min for 500 字, ~20 min for 3,900 字). */
const minutesOf=(chars:number)=>2+chars*.0045;

function decode(bytes:ArrayBuffer){try{return new TextDecoder('utf-8',{fatal:true}).decode(bytes)}catch{return new TextDecoder('gb18030').decode(bytes)}}
/** One file is one script, unless it holds several separated by a line of ---. */
function scriptsIn(name:string,text:string):Item[]{
 const base=name.replace(/\.(txt|md)$/i,'').slice(0,100);
 const parts=text.split(/^\s*-{3,}\s*$/m).map(s=>s.trim()).filter(Boolean);
 if(parts.length<=1)return parts.length?[{title:base,script:parts[0]}]:[];
 return parts.map((s,i)=>({title:(s.split('\n')[0].trim().slice(0,36)||`${base}-${i+1}`),script:s}));
}
const span=(minutes:number)=>minutes<60?`${Math.max(1,Math.round(minutes))} 分钟`:`${Math.floor(minutes/60)} 小时 ${Math.round(minutes%60)} 分钟`;

export function BatchPanel({close,openProject,onCreated}:{close:()=>void;openProject:(id:string)=>void;onCreated:()=>void}){
 const [mode,setMode]=useState<'text'|'audio'>(()=>{try{return localStorage.getItem('jinggao.batchMode')==='audio'?'audio':'text'}catch{return 'text'}});
 const [audios,setAudios]=useState<File[]>([]),[uploading,setUploading]=useState('');
 const [items,setItems]=useState<Item[]>([]),[text,setText]=useState(''),[rows,setRows]=useState<Row[]>([]),[error,setError]=useState(''),[note,setNote]=useState(''),[queueError,setQueueError]=useState(''),[busy,setBusy]=useState(false);
 const [aspect,setAspect]=useState('16:9'),[policy,setPolicy]=useState('video_preferred'),[resolution,setResolution]=useState('1080p');
 const [burn,setBurn]=useState(()=>{try{return localStorage.getItem('jinggao.burn')!=='0'}catch{return true}});
 const [settings,setSettings]=useState<Record<string,any>>({}),[costs,setCosts]=useState<Costs|null>(null);
 const provider=settings.tts_provider||'edge_tts', voices=useVoices(['edge_tts','qwen_tts'].includes(provider)?provider:'edge_tts');
 const workers=Number(settings.video_workers||3);
 const [system,setSystem]=useState<Record<string,any>>({});
 // With the NVIDIA card encoding, more videos at once mostly overlap downloads and model calls.
 // What this computer carries at once (processor and memory); more would make every video slower.
 const limit=Number(system.max_videos||2), recommended=limit;
 const refresh=()=>request('/queue').then((r:Row[])=>{
  // Each project's latest video job; a later retry supersedes the failed run.
  const seen=new Set<string>();setRows(r.filter(j=>j.kind==='produce'&&!seen.has(j.project_id)&&!!seen.add(j.project_id)));setQueueError('')
 }).catch(()=>setQueueError('工作台暂时连接中断，恢复后会自动更新队列。'));
 useEffect(()=>{
  refresh();request('/settings').then(setSettings).catch(()=>{});request('/costs').then(setCosts).catch(()=>{});request('/system').then(setSystem).catch(()=>{});
  const t=setInterval(refresh,2500);return()=>clearInterval(t)
 },[]);
 useEffect(()=>{try{localStorage.setItem('jinggao.burn',burn?'1':'0')}catch{/* private window */}},[burn]);
 useEffect(()=>{try{localStorage.setItem('jinggao.batchMode',mode)}catch{/* private window */}},[mode]);
 const act=async(fn:()=>Promise<void>)=>{setBusy(true);setError('');setNote('');try{await fn()}catch(e){setError((e as Error).message)}finally{setBusy(false)}};
 const change=(key:string,value:string)=>act(async()=>{setSettings(await request('/settings',{[key]:value},'PATCH'))});
 const append=(next:Item[])=>{
  setItems(old=>[...old,...next].slice(0,MAX));
  if(items.length+next.length>MAX)setError(`每批最多 ${MAX} 篇，超过的部分没有加入`);
 };
 const importFiles=(files:FileList|null)=>act(async()=>{
  if(!files)return;
  const chosen=Array.from(files).filter(f=>/\.(txt|md)$/i.test(f.name)).sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name,'zh',{numeric:true}));
  if(!chosen.length)throw new Error('没有找到 TXT / MD 文稿');
  const next:Item[]=[];
  for(const f of chosen)next.push(...scriptsIn(f.name,decode(await f.arrayBuffer())));
  append(next);
 });
 const addAudio=(files:FileList|null)=>{
  if(!files)return;
  const chosen=Array.from(files).filter(f=>AUDIO.test(f.name)).sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name,'zh',{numeric:true}));
  if(!chosen.length){setError('没有找到音频或视频文件（MP3、WAV、M4A、MP4 等）');return}
  setError('');setAudios(old=>[...old,...chosen].slice(0,MAX));
  if(audios.length+chosen.length>MAX)setError(`每批最多 ${MAX} 个，超过的部分没有加入`);
 };
 const submitAudio=()=>act(async()=>{
  // One file per request: big recordings upload one after another, all in the same batch.
  let batch='';const done:string[]=[],failed:string[]=[];
  for(const [i,f] of audios.entries()){
   setUploading(`正在上传 ${i+1}/${audios.length}：${f.name}`);
   const body=new FormData();
   body.append('file',f);body.append('batch',batch);body.append('aspect',aspect);body.append('resolution',resolution);
   body.append('burn_subtitles',burn?'1':'0');body.append('media_policy',policy);
   try{
    const r=await fetch('/api/batch/audio',{method:'POST',body});const d=await r.json();
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'上传失败');
    batch=d.batch;done.push(f.name);
   }catch(e){failed.push(`${f.name}（${(e as Error).message}）`)}
  }
  setUploading('');setAudios([]);onCreated();await refresh();
  setNote(`已加入队列 ${done.length} 条，同时制作 ${workers} 条。可以关掉这个窗口，后台会继续做。`);
  if(failed.length)setError('这些文件没有加入：'+failed.join('；'));
 });
 const chars=items.reduce((n,i)=>n+i.script.length,0);
 const cost=items.reduce((n,i)=>n+costOf(i.script.length),0);
 // Several at once share the CPU and the network: measured gains are well below N times.
 const minutes=items.reduce((n,i)=>n+minutesOf(i.script.length),0)/(1+.4*(workers-1));
 const jobCap=Number(settings.model_job_budget_cny||.1), projectCap=Number(settings.model_project_budget_cny||.1);
 const tooLong=items.filter(i=>costOf(i.script.length)>Math.min(jobCap,projectCap)*.8).length;
 const left=costs?costs.model.daily_limit_cny-costs.model.daily_cny:Infinity;
 const short=cost>left;
 const raise=()=>act(async()=>{
  const day=Math.ceil((costs?costs.model.daily_cny:0)+cost*1.3+.5);
  const one=Math.max(jobCap,Math.ceil(Math.max(...items.map(i=>costOf(i.script.length)))*1.5*10)/10).toFixed(2);
  setSettings(await request('/settings',{model_daily_budget_cny:String(day)+'.00',model_job_budget_cny:one,model_project_budget_cny:one},'PATCH'));
  setCosts(await request('/costs'));setNote(`已调高：今日模型上限 ¥${day}，单条上限 ¥${one}`);
 });
 const submit=()=>act(async()=>{
  const r=await request('/batch',{projects:items.map(i=>({...i,aspect,media_policy:policy,burn_subtitles:burn})),resolution,workers});
  setItems([]);onCreated();await refresh();setNote(`已加入队列 ${r.projects.length} 篇，同时制作 ${workers} 条。可以关掉这个窗口，进度会在后台继续。`);
 });
 const live=rows.filter(r=>r.status==='running'||r.status==='queued');
 const failed=rows.filter(r=>r.status==='failed'||r.status==='interrupted');
 const cancelled=rows.filter(r=>r.status==='cancelled');
 const count=(s:string)=>rows.filter(r=>r.status===s).length;
 const sorted=[...rows].sort((a,b)=>(order[a.status]??6)-(order[b.status]??6)||(a.status==='queued'?a.created-b.created:b.updated-a.updated)).slice(0,MAX);
 const cancelAll=()=>{if(confirm(`取消全部 ${live.length} 个排队和制作中的视频？已经做好的不受影响。`))act(async()=>{const r=await request('/queue/cancel',{});await refresh();setNote(`已取消：排队 ${r.queued} 个，正在制作的 ${r.running} 个会在当前步骤结束后停下。`)})};
 const retryAll=()=>act(async()=>{const r=await request('/queue/retry',{});await refresh();setNote(`已重新排队 ${r.retried} 个，会从各自中断的地方继续。`)});
 const restartCancelled=()=>act(async()=>{const r=await request('/queue/retry',{cancelled:true});await refresh();setNote(`已重新排队 ${r.retried} 个，会从各自停下的地方继续。`)});
 return <div className="modal-backdrop"><section className="modal batch-modal" role="dialog" aria-modal="true" aria-label="批量制作"><div className="modal-heading"><div><h2>批量制作</h2></div><button className="icon-button" aria-label="关闭批量制作" onClick={close}><X/></button></div><div className="modal-content">
  <div className="batch-mode">
   <button className={mode==='text'?'on':''} onClick={()=>setMode('text')}><FileText size={22}/><strong>文稿批量</strong><small>一篇文稿做一条视频，自动配音</small></button>
   <button className={mode==='audio'?'on':''} onClick={()=>setMode('audio')}><AudioLines size={22}/><strong>音频批量</strong><small>已经配好音（如百度一镜导出的），一个音频做一条</small></button>
  </div>
  <div className="batch-step"><h3><b>1</b>{mode==='text'?'选文稿':'选音频'}</h3>
  {mode==='text'?<>
   <p>一个 TXT 文件就是一条视频，标题就是文件名。在选文件的窗口里按 Ctrl+A 可以一次全选。</p>
   <div className="batch-import">
    <label className="secondary"><Upload size={18}/>选几个 TXT 文件<input hidden type="file" multiple accept=".txt,.md,text/plain" disabled={busy} onChange={e=>{importFiles(e.target.files);e.target.value=''}}/></label>
    <label className="secondary"><FolderOpen size={18}/>选整个文件夹<input hidden type="file" multiple disabled={busy} {...{webkitdirectory:''}} onChange={e=>{importFiles(e.target.files);e.target.value=''}}/></label>
   </div>
   <details className="batch-paste"><summary>也可以直接粘贴文稿</summary>
    <label>多篇之间单独一行写 ---<textarea rows={4} value={text} onChange={e=>setText(e.target.value)} placeholder="每篇文稿生成一个独立视频"/></label>
    <button className="secondary small" disabled={!text.trim()||busy} onClick={()=>{append(scriptsIn('粘贴',text));setText('')}}><Plus size={16}/>加入本次列表</button>
   </details>
   {items.length>0&&<div className="batch-list">
    <div className="batch-list-head"><strong>已选 {items.length} 篇 · {chars.toLocaleString()} 字</strong><button className="text-link" onClick={()=>setItems([])}><Trash2 size={12}/>清空</button></div>
    {items.map((item,i)=><div className="batch-item" key={i}><span>{i+1}</span><input aria-label={`第 ${i+1} 篇标题`} value={item.title} maxLength={100} onChange={e=>setItems(items.map((v,j)=>j===i?{...v,title:e.target.value}:v))}/><small>{item.script.length} 字</small><button className="icon-button" aria-label={`移除第 ${i+1} 篇`} onClick={()=>setItems(items.filter((_,j)=>j!==i))}><X size={16}/></button></div>)}
   </div>}
  </>:<>
   <p>一个音频做一条视频，标题就是文件名。软件会自己听出文字、配画面、按原声加字幕，不再配音。MP3、WAV、M4A 都行；百度一镜导出的 MP4 视频也可以直接放进来，只用它的声音。</p>
   <div className="batch-import">
    <label className="secondary"><Upload size={18}/>选几个音频文件<input hidden type="file" multiple accept="audio/*,video/*,.mp3,.wav,.m4a,.aac,.flac,.mp4,.mov" disabled={busy} onChange={e=>{addAudio(e.target.files);e.target.value=''}}/></label>
    <label className="secondary"><FolderOpen size={18}/>选整个文件夹<input hidden type="file" multiple disabled={busy} {...{webkitdirectory:''}} onChange={e=>{addAudio(e.target.files);e.target.value=''}}/></label>
   </div>
   {audios.length>0&&<div className="batch-list">
    <div className="batch-list-head"><strong>已选 {audios.length} 个音频 · {megabytes(audios.reduce((n,f)=>n+f.size,0))}</strong><button className="text-link" disabled={busy} onClick={()=>setAudios([])}><Trash2 size={12}/>清空</button></div>
    {audios.map((f,i)=><div className="batch-item" key={i}><span>{i+1}</span><input aria-label={`第 ${i+1} 个音频`} value={f.name.replace(/\.[^.]+$/,'')} readOnly/><small>{megabytes(f.size)}</small><button className="icon-button" aria-label={`移除第 ${i+1} 个`} disabled={busy} onClick={()=>setAudios(audios.filter((_,j)=>j!==i))}><X size={16}/></button></div>)}
   </div>}
  </>}
  </div>
  <div className="batch-step"><h3><b>2</b>设置<small>一般不用改</small></h3>
  <div className="field-grid">
   <label>同时制作<select value={String(Math.min(workers,limit))} onChange={e=>change('video_workers',e.target.value)}>{[1,2,3,4,5,6].map(n=><option key={n} value={n} disabled={n>limit}>{n} 条{n===recommended?'（这台电脑最多）':n>limit?'（这台电脑带不动）':''}</option>)}</select>
    <small className="field-hint">{system.cores?`这台电脑 ${system.cores} 个线程、${Math.round(system.memory_gb||0)} GB 内存，最多同时 ${limit} 条；再多每条都会变慢，第一条要等很久才出来。`:''}</small></label>
   <label>清晰度<select value={resolution} onChange={e=>setResolution(e.target.value)}><option value="1080p">1080p 高清</option><option value="720p">720p · 更快</option></select></label>
   <label>画面比例<select value={aspect} onChange={e=>setAspect(e.target.value)}><option value="16:9">横屏 16:9</option><option value="9:16">竖屏 9:16</option></select></label>
   <label>素材组合<select value={policy} onChange={e=>setPolicy(e.target.value)}><option value="video_preferred">至少 80% 实拍视频</option><option value="video_only">全部实拍视频</option><option value="mixed">图片与视频混合</option></select></label>
   {mode==='text'&&voices&&voices.voices.length>0&&<label>配音音色<select value={settings[voices.voice_key]||voices.voice} onChange={e=>change(voices.voice_key,e.target.value)}>{voices.voices.map(v=><option key={v.id} value={v.id}>{v.name} · {v.note}</option>)}</select></label>}
   <label className="checkbox burn-toggle"><input type="checkbox" checked={burn} onChange={e=>setBurn(e.target.checked)}/>画面上显示字幕<small>{burn?'直接出片带字幕':'不带字幕，另附 SRT（进剪映用）'}</small></label>
  </div>
  </div>
  <div className="batch-step"><h3><b>3</b>开始</h3>
  {mode==='text'&&items.length>0&&<p className="muted-note">预计用时约 {span(minutes)}（同时 {workers} 条，粗略估计；取决于电脑性能、网速和 B 站限流），模型费约 ¥{cost.toFixed(2)}，免费配音不收费。</p>}
  {mode==='audio'&&audios.length>0&&<p className="muted-note">模型费约 ¥{(audios.length*.03).toFixed(2)}（每条约 0.03 元），不收配音费。</p>}
  {mode==='text'&&items.length>0&&(short||tooLong>0)&&<div className="inline-note warning">
   {short&&<>今日模型费用上限还剩 ¥{Math.max(0,left).toFixed(2)}，这一批约需 ¥{cost.toFixed(2)}，不调高的话做到一半会停下。</>}
   {tooLong>0&&<> 有 {tooLong} 篇文稿较长，可能超过单条 ¥{Math.min(jobCap,projectCap).toFixed(2)} 的上限。</>}
   <button className="text-link" disabled={busy} onClick={raise}>一键调高上限</button>
  </div>}
  {mode==='text'?<button className="primary big" disabled={busy||!items.length||items.some(i=>!i.title.trim()||i.script.length>30000)} onClick={submit}>{busy?'正在提交…':items.length?`开始制作 ${items.length} 条视频`:'先在第 1 步选文稿'}</button>
   :<button className="primary big" disabled={busy||!audios.length} onClick={submitAudio}>{busy?(uploading||'正在提交…'):audios.length?`开始制作 ${audios.length} 条视频`:'先在第 1 步选音频'}</button>}
  </div>
  {error&&<p role="alert" className="candidate-warning">{error}</p>}
  {note&&<p role="status" className="muted-note">{note}</p>}
  {queueError&&<p role="status" className="candidate-warning">{queueError}</p>}
  <div className="queue-head"><h3>制作队列</h3>
   {rows.length>0&&<span>制作中 {count('running')} · 排队 {count('queued')} · 已完成 {count('completed')}{failed.length?` · 失败 ${failed.length}`:''}</span>}
   <div>{failed.length>0&&<button className="secondary small" disabled={busy} onClick={retryAll}><RotateCcw size={14}/>失败的全部重试</button>}
   {cancelled.length>0&&<button className="secondary small" disabled={busy} onClick={restartCancelled}><RotateCcw size={14}/>取消的全部重新开始</button>}
   {live.length>0&&<button className="secondary small danger" disabled={busy} onClick={cancelAll}><Square size={13}/>全部取消</button>}</div>
  </div>
  <FolderBar label="做好的视频自动保存在（每条一个 MP4，文件名就是标题）"/>
  {!rows.length&&<p>还没有整片制作任务。</p>}
  {sorted.map(row=><div className="queue-row" key={row.id}><div><button className="text-link" onClick={()=>{close();openProject(row.project_id)}}>{row.title}</button><span>{names[row.status]||row.status}</span></div>{row.status!=='completed'&&row.status!=='cancelled'&&<progress value={row.progress} max={100}/>}{row.status==='completed'&&row.video&&<div className="queue-file"><span title={row.video.saved_to}><CheckCircle2 size={13}/>{row.video.saved_to?'已保存为 '+fileName(row.video.saved_to):'已完成（自动保存已关闭，点标题查看）'}</span><VideoFileActions pid={row.project_id} path={row.video.saved_to||undefined} compact/><button className="text-link" onClick={()=>{close();openProject(row.project_id)}}>在这里看成片</button></div>}<p>{row.message}{row.error&&` · ${row.error}`}</p>{['failed','needs_review','cancelled','interrupted'].includes(row.status)&&<button className="secondary small" disabled={busy} onClick={()=>act(async()=>{await request(`/jobs/${row.id}/retry`,{});await refresh()})}>从未完成处继续</button>}{['running','queued'].includes(row.status)&&<button className="text-link" disabled={busy} onClick={()=>act(async()=>{await request(`/jobs/${row.id}/cancel`,{});await refresh()})}>取消这一条</button>}</div>)}
 </div><div className="modal-footer"><small>可以关掉这个窗口，后台会继续做；做好的视频自动存到「制作队列」上方写的文件夹。</small><button className="secondary" onClick={close}>关闭</button></div></section></div>;
}
