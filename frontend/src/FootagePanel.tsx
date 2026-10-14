import {useCallback, useEffect, useState} from 'react';
import {Database, Download, Loader2, Upload, X} from 'lucide-react';
import type {Job} from './types';

type Stats = {sources:number;described:number;moments:number;described_moments:number;file_mb:number;
 topics:{topic:string;sources:number}[];packs:Record<string,number>;job:Job|null;
 report?:{found:number;indexed:number;described_frames:number;failed:number;blocked:boolean}|null};
async function call<T>(path:string,method='GET',body?:unknown):Promise<T>{
 const r=await fetch('/api'+path,{method,headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)});
 const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'请求无效');return data;
}
const live=(j?:Job|null)=>!!j&&['queued','running'].includes(j.status);

/** The local footage index: what it holds, and growing it by topic packs or keywords (no video is downloaded). */
export function FootagePanel({close}:{close:()=>void}) {
 const [stats,setStats]=useState<Stats|null>(null),[error,setError]=useState(''),[note,setNote]=useState('');
 const [packs,setPacks]=useState<string[]>([]),[keywords,setKeywords]=useState(''),[pages,setPages]=useState(1);
 const load=useCallback(()=>call<Stats>('/footage-index').then(setStats).catch(e=>setError((e as Error).message)),[]);
 useEffect(()=>{load()},[load]);
 useEffect(()=>{if(!live(stats?.job))return;const t=setInterval(load,2000);return()=>clearInterval(t)},[stats?.job?.status,stats?.job?.id,load]);
 const words=keywords.split(/\n|，|,/).map(s=>s.trim()).filter(s=>s.length>=2);
 const queries=packs.reduce((n,p)=>n+(stats?.packs[p]||0),0)+words.length;
 // About 15 new videos a page of results, 10 frames each, 24 frames to a sheet at about 0.0015 元.
 const estimate=queries*pages*15*10/24*0.0015;
 const count=(name:string)=>stats?.topics.find(t=>t.topic===name)?.sources||0;
 const start=async()=>{setError('');setNote('');try{await call('/footage-index/build','POST',{packs,keywords:words,pages});await load()}catch(e){setError((e as Error).message)}};
 const job=stats?.job;
 return <div className="modal-backdrop"><section className="modal footage-modal" role="dialog" aria-modal="true" aria-label="素材库">
  <div className="modal-heading"><div><h2><Database size={19}/> 素材库</h2></div><button className="icon-button" aria-label="关闭素材库" onClick={close}><X size={20}/></button></div>
  <div className="modal-content">
   <p>只存 B 站视频的标题、标签和预览帧的画面描述，不下载视频；出片时先从这里找画面（能直接定位到对应的那几秒），找不到再现场搜索，最后只下载选中的片段。每做一条视频，用到的搜索结果也会自动收进来。</p>
   {stats&&<div className="footage-stats">
    <div><strong>{stats.sources.toLocaleString()}</strong><span>个视频</span></div>
    <div><strong>{stats.described.toLocaleString()}</strong><span>已描述画面</span></div>
    <div><strong>{stats.described_moments.toLocaleString()}</strong><span>帧可检索</span></div>
    <div><strong>{stats.file_mb}</strong><span>MB</span></div>
   </div>}
   {live(job)?<div className="footage-job"><Loader2 className="spin" size={16}/><span>{job!.status==='queued'?'排队中，前面的视频做完后开始':job!.message}</span>
     <div className="progress-track big"><i style={{width:Math.max(2,job!.progress)+'%'}}/></div>
     <button className="secondary small" disabled={!!job!.cancel} onClick={()=>call('/jobs/'+job!.id+'/cancel','POST').then(load)}>停止</button></div>
    :job&&<p className="muted-note">上次扩充：{job.message}{job.status==='failed'&&job.error?` · ${job.error}`:''}</p>}
   <h3>按题材扩充 <button className="link" onClick={()=>setPacks(p=>p.length?[]:Object.keys(stats?.packs||{}))}>{packs.length?'清空':'全选'}</button></h3>
   <div className="pack-grid">{Object.entries(stats?.packs||{}).map(([name,n])=><label key={name} className={'pack'+(packs.includes(name)?' on':'')}>
    <input type="checkbox" checked={packs.includes(name)} onChange={e=>setPacks(p=>e.target.checked?[...p,name]:p.filter(x=>x!==name))}/>
    <strong>{name}</strong><small>{n} 个关键词 · 已收 {count(name)} 个视频</small></label>)}</div>
   <label>自定义关键词（每行一个，例如「宋朝 粮仓」「港口 吊车 航拍」）<textarea rows={4} value={keywords} onChange={e=>setKeywords(e.target.value)} placeholder={'宋朝 粮仓\n港口 吊车 航拍'}/></label>
   <label className="inline">每个关键词取<select value={pages} onChange={e=>setPages(Number(e.target.value))}><option value={1}>约 20 个结果</option><option value={2}>约 40 个结果</option><option value={3}>约 60 个结果</option></select></label>
   {error&&<div className="message error" role="alert">{error}</div>}
   {note&&<p className="muted-note">{note}</p>}
   <div className="footage-files">
    <a className="secondary small" href="/api/footage-index/file"><Download size={14}/>导出素材库文件</a>
    <label className="secondary small"><Upload size={14}/>导入素材库文件<input type="file" hidden accept=".sqlite,application/octet-stream" onChange={async e=>{const f=e.target.files?.[0];e.target.value='';if(!f)return;setError('');const body=new FormData();body.append('file',f);try{const r=await call<Stats&{added:number}>('/footage-index/import','POST',body);setNote(`已导入 ${r.added} 个视频`);await load()}catch(err){setError((err as Error).message)}}}/></label>
    <small>导出的文件可以复制到另一台电脑再导入，不用重新建。扩充期间新提交的视频要排队，建议空闲时扩充。</small>
   </div>
  </div>
  <div className="modal-footer"><span className="estimate">{queries?`${queries} 个关键词 · 预计画面描述费用约 ¥${Math.max(0.01,estimate).toFixed(2)}（单独记账，每日上限见设置）`:'选择题材包或填写关键词'}</span>
   <button className="secondary" onClick={close}>关闭</button>
   <button className="primary" disabled={!queries||live(job)} onClick={start}><Database size={16}/>开始扩充</button></div>
 </section></div>;
}
