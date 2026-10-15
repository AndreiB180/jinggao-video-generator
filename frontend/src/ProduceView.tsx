import {useEffect, useRef, useState} from 'react';
import {Check, Clapperboard, Download, FolderOpen, Loader2, Plus, RefreshCw, Subtitles, TriangleAlert} from 'lucide-react';
import type {Job, Project} from './types';

const fileUrl=(pid:string,path:string)=>`/api/projects/${pid}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
const clock=(s:number)=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
const live=(j?:Job)=>!!j&&['queued','running'].includes(j.status);

/** Which numbered stage ("2/4 生成完整配音 · …") a produce job is in; the alignment message sits in stage 2. */
function stageOf(message:string,last:number){
 const m=/^(\d)\/4/.exec(message);
 if(m)return Number(m[1]);
 if(/对齐|本地识别配音/.test(message))return Math.max(last,2);
 return last;
}

/** A project's main page: where its video is being made, or the finished video with its files. */
export function ProduceView({project,jobs,act,refine,again}:{project:Project;jobs:Job[];act:(path:string,body?:unknown)=>Promise<void>;refine:()=>void;again:()=>void}) {
 const job=jobs.find(j=>j.kind==='produce')||jobs[0];
 const audio=!!project.source_audio;
 const last=useRef(0);
 const [now,setNow]=useState(Date.now()/1000);
 useEffect(()=>{if(!live(job))return;const t=setInterval(()=>setNow(Date.now()/1000),1000);return()=>clearInterval(t)},[job?.id,job?.status]);
 const [opened,setOpened]=useState('');
 const video=project.exports[0];
 const steps=[audio?'识别文字':'整理文案','规划分镜',audio?'对齐原声':'生成配音','匹配画面','合成视频'];
 if(live(job)){
  const stage=stageOf(job!.message,last.current);last.current=stage;
  const created=job!.created;
  return <div className="produce">
   <div className="produce-card">
    <h2>正在生成「{project.title}」</h2>
    <ol className="steps">{steps.map((name,i)=><li key={name} className={i<stage?'done':i===stage?'now':''}><span>{i<stage?<Check size={14}/>:i+1}</span>{name}</li>)}</ol>
    <div className="progress-track big"><i style={{width:Math.max(2,job!.progress)+'%'}}/></div>
    <p className="produce-message">{job!.status==='queued'?'排队中，前一个任务完成后开始':job!.message.replace(/^\d\/4\s*/,'')}</p>
    <div className="produce-foot"><span>{Math.round(job!.progress)}%{created?` · 已用 ${clock(Math.max(0,now-created))}`:''}</span><span>可以关掉这个页面，后台会继续；完成后自动保存到成片文件夹。</span>
     <button className="secondary small" disabled={!!job!.cancel} onClick={()=>act('/jobs/'+job!.id+'/cancel')}>{job!.cancel?'正在取消…':'取消'}</button></div>
   </div>
  </div>;
 }
 if(job&&['failed','interrupted','needs_review','cancelled'].includes(job.status)&&(!video||(video.created<(job.updated||0)))){
  return <div className="produce"><div className="produce-card failed">
   <h2><TriangleAlert size={20}/>{job.status==='cancelled'?'已取消':'没有完成'}</h2>
   <p>{job.message}</p>{job.error&&<p className="muted-note">{job.error}</p>}
   <div className="produce-actions"><button className="primary" onClick={()=>act('/jobs/'+job.id+'/retry')}><RefreshCw size={16}/>从中断处继续</button><button className="secondary" onClick={refine}>进入精修</button></div>
  </div></div>;
 }
 if(!video){
  return <div className="produce"><div className="produce-card">
   <h2>还没有成片</h2><p>这个项目还没有生成视频。可以在「文稿」里检查文案后点右上角「生成视频」，或进入分镜逐镜精修。</p>
   <div className="produce-actions"><button className="secondary" onClick={refine}>进入精修</button></div>
  </div></div>;
 }
 const srt=video.path.replace(/[^/]+$/,'subtitles.srt');
 const burned=video.subtitles!==false;
 const other=project.exports.find(e=>(e.subtitles!==false)!==burned);
 const saved=video.saved_to;
 const tall=project.aspect==='9:16';
 return <div className="produce done">
  <div className={'produce-video'+(tall?' tall':'')}><video key={video.id} controls src={fileUrl(project.id,video.path)}/></div>
  <div className="produce-side">
   <h2>{project.title}</h2>
   <p className="meta">{clock(video.duration)} · {video.width}×{video.height} · {project.shots.length} 个镜头 · {burned?'带字幕':'无字幕'} · {new Date(video.created*1000).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}</p>
   {saved?<p className="saved"><Check size={14}/>已保存到 <span title={saved}>{saved}</span></p>:null}
   <div className="produce-actions column">
    <a className="primary" href={fileUrl(project.id,video.path)} download={project.title+'.mp4'}><Download size={16}/>下载视频</a>
    <a className="secondary" href={fileUrl(project.id,srt)} download={project.title+'.srt'}><Subtitles size={16}/>下载字幕 SRT</a>
    {other?<a className="secondary" href={fileUrl(project.id,other.path)} download={project.title+(burned?'（无字幕）':'')+'.mp4'}><Download size={16}/>下载{burned?'无字幕':'带字幕'}版</a>
     :<button className="secondary" disabled={live(job)} onClick={()=>act('/projects/'+project.id+'/jobs',{kind:'render',resolution:video.height>=1000||video.width>=1900?'1080p':'720p',subtitles:!burned})}>
      <Subtitles size={16}/>再导出一版{burned?'无字幕的（给剪映用）':'带字幕的'}</button>}
    <button className="secondary" onClick={async()=>{try{const r=await fetch('/api/output-folder/open',{method:'POST'});const d=await r.json();setOpened(d.path||'')}catch{setOpened('')}}}><FolderOpen size={16}/>打开成片文件夹</button>
    {opened&&<small className="muted-note">已打开 {opened}</small>}
   </div>
   <p className="muted-note">字幕已烧录在视频里；SRT 只在上传平台需要外挂字幕时使用，播放时不要再加载，否则会出现两行字幕。</p>
   <div className="produce-actions">
    <button className="secondary" onClick={refine}><Clapperboard size={16}/>精修画面</button>
    <button className="secondary" onClick={again}><Plus size={16}/>再做一条</button>
   </div>
   {job?.status==='running'&&<p><Loader2 className="spin" size={14}/></p>}
  </div>
 </div>;
}
