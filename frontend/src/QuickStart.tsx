import {useEffect, useRef, useState} from 'react';
import {AudioLines, FileText, FolderOpen, Loader2, RotateCcw, Upload, WandSparkles, X} from 'lucide-react';
import {VoicePicker, useVoices} from './VoicePicker';
import {SubtitleSettings} from './SubtitleSettings';

type Settings = Record<string,any>;
async function call<T>(path:string,method='GET',body?:unknown):Promise<T>{
 const r=await fetch('/api'+path,{method,headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)});
 const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'请求内容无效，请检查输入');return data;
}
const plain=(s:string)=>s.replace(/\s+/g,'');
const clock=(sec:number)=>`${Math.floor(sec/60)} 分 ${String(Math.round(sec%60)).padStart(2,'0')} 秒`;
const DRAFT='jinggao.quick.draft';

/** The home screen: paste a script (or drop a recording), pick a voice and subtitle look, generate. */
export function QuickStart({started,openNews}:{started:(projectId:string)=>void;openNews:()=>void}) {
 const [mode,setMode]=useState<'text'|'audio'>('text');
 const [script,setScript]=useState(()=>{try{return localStorage.getItem(DRAFT)||''}catch{return ''}});
 const [title,setTitle]=useState(''),[file,setFile]=useState<File|null>(null),[audioSeconds,setAudioSeconds]=useState(0);
 const [aspect,setAspect]=useState<'16:9'|'9:16'>('16:9'),[resolution,setResolution]=useState('1080p');
 const [settings,setSettings]=useState<Settings>({}),[folder,setFolder]=useState('');
 const [busy,setBusy]=useState(''),[error,setError]=useState(''),[original,setOriginal]=useState<string|null>(null);
 const [dragging,setDragging]=useState(false);
 const input=useRef<HTMLInputElement>(null);
 useEffect(()=>{call<Settings>('/settings').then(setSettings).catch(()=>{});call<{path:string}>('/output-folder').then(f=>setFolder(f.path)).catch(()=>{})},[]);
 useEffect(()=>{try{localStorage.setItem(DRAFT,script)}catch{/* private window */}},[script]);
 const change=(key:string,value:string)=>{setSettings(s=>({...s,[key]:value}));call('/settings','PATCH',{[key]:value}).catch(e=>setError((e as Error).message))};
 const pick=(f:File|undefined|null)=>{
  if(!f)return;
  if(!/^(audio|video)\//.test(f.type)&&!/\.(mp3|wav|m4a|aac|flac|ogg|mp4|mov)$/i.test(f.name)){setError('请选择音频文件（MP3、WAV、M4A 等）');return}
  setError('');setFile(f);setAudioSeconds(0);
  const url=URL.createObjectURL(f),probe=new Audio();probe.preload='metadata';
  probe.onloadedmetadata=()=>{setAudioSeconds(probe.duration||0);URL.revokeObjectURL(url)};probe.onerror=()=>URL.revokeObjectURL(url);probe.src=url;
 };
 const chars=plain(script).length;
 const speed=Number(settings.tts_speed||'1.1');
 const seconds=mode==='text'?chars/(4.2*speed):audioSeconds;
 const provider=settings.tts_provider||'edge_tts';
 const voices=useVoices(provider);
 const voiceCost=mode==='text'?chars/1000*(voices?.price_per_1000_chars??0):0;
 const modelCost=seconds?Math.min(0.12,0.012+seconds/60*0.006):0;
 const ready=mode==='text'?chars>=10:!!file;
 const rewrite=async()=>{
  setBusy('rewrite');setError('');
  try{const r=await call<{script:string;title:string}>('/rewrite','POST',{text:script,title,aspect});setOriginal(script);setScript(r.script);if(!title&&r.title)setTitle(r.title)}
  catch(e){setError((e as Error).message)}finally{setBusy('')}
 };
 const start=async()=>{
  setBusy('start');setError('');
  try{
   const first=plain(script).split(/[。！？!?，,；;]/)[0]?.slice(0,24);
   const name=title.trim()||(mode==='text'?first:file?.name.replace(/\.[^.]+$/,'').slice(0,40))||'新视频';
   const clean=(t:string)=>t.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g,'');   // broken characters from a bad copy
   const p=await call<{id:string}>('/projects','POST',{title:clean(name),script:mode==='text'?clean(script).trim():'',aspect});
   if(mode==='audio'&&file){const body=new FormData();body.append('file',file);body.append('role','source_audio');await call('/projects/'+p.id+'/upload','POST',body)}
   await call('/projects/'+p.id+'/jobs','POST',{kind:'produce',resolution});
   if(mode==='text'){setScript('');setOriginal(null)}else setFile(null);
   setTitle('');started(p.id);
  }catch(e){setError((e as Error).message)}finally{setBusy('')}
 };
 return <div className="quick">
  <div className="quick-main">
   <h1>{mode==='text'?'粘贴文案，一键成片':'上传口播，一键成片'}</h1>
   <p className="lead">{mode==='text'?'自动配音、配画面、加字幕':'自动识别文字，按原声配画面、加字幕'}，完成后自动保存到成片文件夹。</p>
   <div className="segmented big" role="tablist">
    <button role="tab" aria-selected={mode==='text'} className={mode==='text'?'on':''} onClick={()=>setMode('text')}><FileText size={16}/>文案成片</button>
    <button role="tab" aria-selected={mode==='audio'} className={mode==='audio'?'on':''} onClick={()=>setMode('audio')}><AudioLines size={16}/>音频成片</button>
   </div>
   {mode==='text'?<div className="script-box">
    <textarea value={script} onChange={e=>setScript(e.target.value)} placeholder={'在这里粘贴口播文案……\n\n没有标点、带序号或是从网页复制的文字都可以，生成前会自动整理。'} aria-label="口播文案"/>
    <div className="script-foot">
     <span>{chars} 字{chars>0&&<> · 约 {clock(seconds)}</>}</span>
     {original!==null&&<button className="link" onClick={()=>{setScript(original);setOriginal(null)}}><RotateCcw size={13}/>恢复原稿</button>}
     <button className="link" disabled={chars<20||!!busy} onClick={rewrite} title="换一种说法改写成口播稿，并对照原文核对事实">{busy==='rewrite'?<Loader2 className="spin" size={13}/>:<WandSparkles size={13}/>}AI 改写</button>
     <button className="link" onClick={openNews}>从新闻选题</button>
     {script&&<button className="link" onClick={()=>{setScript('');setOriginal(null)}}><X size={13}/>清空</button>}
    </div>
   </div>:<div className={'drop-zone'+(dragging?' over':'')+(file?' has-file':'')} onClick={()=>input.current?.click()} onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);pick(e.dataTransfer.files[0])}} role="button" tabIndex={0} onKeyDown={e=>{if(e.key==='Enter'||e.key===' ')input.current?.click()}}>
    <input ref={input} type="file" hidden accept="audio/*,.m4a,.mp3,.wav,.aac,.flac,video/mp4" onChange={e=>{pick(e.target.files?.[0]);e.target.value=''}}/>
    {file?<><AudioLines size={30}/><strong>{file.name}</strong><span>{(file.size/1048576).toFixed(1)} MB{audioSeconds>0&&<> · {clock(audioSeconds)}</>}</span><button className="link" onClick={e=>{e.stopPropagation();setFile(null)}}>换一个</button></>
     :<><Upload size={30}/><strong>拖入音频，或点击选择</strong><span>MP3 / WAV / M4A，3 秒到 60 分钟。自动识别文字，用原声配画面和字幕。</span></>}
   </div>}
   <div className="go-bar">
    <label className="title-field">标题<span>（可不填，默认用第一句话）</span><input value={title} maxLength={100} onChange={e=>setTitle(e.target.value)} placeholder="视频标题"/></label>
    <button className="primary go" disabled={!ready||!!busy} onClick={start}>{busy==='start'?<><Loader2 className="spin" size={18}/>正在提交…</>:'开始生成'}</button>
   </div>
   {error&&<div className="message error" role="alert">{error}</div>}
   <p className="estimate">{ready&&seconds>0?<>视频约 {clock(seconds)} · 预计费用约 ¥{(voiceCost+modelCost).toFixed(2)}{mode==='text'?(voiceCost>0?`（其中配音 ¥${voiceCost.toFixed(2)}）`:'（配音免费）'):''} · 预计用时 {Math.max(2,Math.round((60+seconds*1.2)/60))} 分钟左右</>:mode==='text'?'粘贴至少 10 个字的文案':'选择一个音频文件'}</p>
  </div>
  <aside className="quick-side">
   <section className="card"><h3>画面比例</h3>
    <div className="aspect-pick">{([['16:9','横屏','西瓜 / B 站 / 头条'],['9:16','竖屏','抖音 / 快手 / 视频号']] as const).map(([value,name,where])=><button key={value} className={aspect===value?'on':''} onClick={()=>setAspect(value)}><i className={value==='16:9'?'wide':'tall'}/><span><strong>{name} {value}</strong><small>{where}</small></span></button>)}</div>
   </section>
   {mode==='text'&&<section className="card"><h3>配音</h3>
    {!['edge_tts','qwen_tts'].includes(provider)?<p className="muted-note">当前在设置里选择了其他配音方式。</p>:<>
     <VoicePicker compact provider={provider} values={settings} change={change}/>
     <p className="muted-note">{provider==='edge_tts'?'免费配音（微软神经网络音色），不收费。':'千问配音，约 ¥0.16 / 千字。'}可在设置里切换。</p></>}
   </section>}
   <section className="card"><h3>字幕</h3><SubtitleSettings compact values={settings} change={change}/></section>
   <section className="card quiet">
    <label className="inline">清晰度<select value={resolution} onChange={e=>setResolution(e.target.value)}><option value="1080p">1080p 高清</option><option value="720p">720p 更快</option></select></label>
    {folder&&<p className="folder-line"><FolderOpen size={14}/><span title={folder}>成片保存到 {folder}</span></p>}
   </section>
  </aside>
 </div>;
}
