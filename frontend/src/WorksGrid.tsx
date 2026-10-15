import {AudioLines, Film, Loader2, Plus, TriangleAlert} from 'lucide-react';
import type {Summary} from './types';
import {FolderBar, VideoFileActions, fileName} from './FileActions';

const fileUrl=(pid:string,path:string)=>`/api/projects/${pid}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
const clock=(s:number)=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

/** Every project as a card: its latest video (first frame), length, where the file is and whether it is still being made. */
export function WorksGrid({projects,open,create}:{projects:Summary[];open:(id:string)=>void;create:()=>void}) {
 return <div className="works">
  <div className="works-head"><h1>我的作品</h1><span>{projects.length} 个</span><button className="primary small" onClick={create}><Plus size={15}/>新建</button></div>
  <FolderBar/>
  {!projects.length&&<div className="empty-state"><Film size={40}/><h2>还没有作品</h2><p>回到「一键成片」，粘贴一段文案就能开始。</p></div>}
  <div className="works-grid">{projects.map(p=>{
   const running=p.job&&['queued','running'].includes(p.job.status);
   const failed=p.job&&['failed','interrupted','needs_review'].includes(p.job.status)&&!running;
   const saved=p.video?.saved_to;
   return <div key={p.id} className="work-card">
    <button className={'work-thumb'+(p.aspect==='9:16'?' tall':'')} onClick={()=>open(p.id)} title={'打开「'+p.title+'」'}>
     {p.video?<video src={fileUrl(p.id,p.video.path)+'#t=1.5'} preload="metadata" muted playsInline/>:<Film size={28}/>}
     {p.video&&<span className="len">{clock(p.video.duration||0)}</span>}
     {running&&<span className="badge live"><Loader2 className="spin" size={12}/>{p.job!.status==='queued'?'排队中':Math.round(p.job!.progress)+'%'}</span>}
     {failed&&<span className="badge bad"><TriangleAlert size={12}/>未完成</span>}
    </button>
    <button className="work-title" onClick={()=>open(p.id)} title={p.title}><strong>{p.title}</strong></button>
    <small>{p.audio_source&&<><AudioLines size={11}/> 音频成片 · </>}{p.shot_count} 个镜头 · {new Date(p.updated*1000).toLocaleDateString('zh-CN',{month:'numeric',day:'numeric'})}</small>
    {p.video&&<div className="work-file">
     <span title={saved||'自动保存已关闭：成片在项目里，可在作品页下载'}>{saved?fileName(saved):'未自动保存'}</span>
     <VideoFileActions pid={p.id} path={saved} exportId={p.video.id} compact/>
    </div>}
   </div>})}</div>
 </div>;
}
