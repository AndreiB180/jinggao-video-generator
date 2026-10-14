import {useEffect,useMemo,useRef,useState} from 'react';
import {ChevronLeft,ChevronRight,Pause,Pencil,Play,X} from 'lucide-react';
import type {Project} from './types';
import {captionCss,FramedVideo,framing,pct,projectFile,useSubtitleFont,type CaptionStyle} from './ShotPreview';

type FilmShot={id:string;start:number;duration:number;segments:[number,number,string][];audio_path:string|null;overlay_label:string};
type Film={total:number;audio_path:string|null;aligned:boolean;alignment_note:string;alignment:string;shots:FilmShot[];style?:CaptionStyle};
const clockText=(n:number)=>`${Math.floor(n/60).toString().padStart(2,'0')}:${Math.floor(n%60).toString().padStart(2,'0')}`;

/** Plays every shot back to back on the export's timeline (same cuts, crops, subtitles and narration) before rendering. */
export function FilmPreview({project,close,edit}:{project:Project;close:()=>void;edit:(shotId:string)=>void}){
 const [film,setFilm]=useState<Film|null>(null),[error,setError]=useState(''),[t,setT]=useState(0),[playing,setPlaying]=useState(false);
 const audio=useRef<HTMLAudioElement>(null),music=useRef<HTMLAudioElement>(null),videos=useRef(new Map<string,HTMLVideoElement>());
 // Wall clock used whenever no narration is playing; `waiting` holds it while a shot's own narration loads.
 const clock=useRef({t:0,at:0}),time=useRef(0),waiting=useRef(false);
 const family=useSubtitleFont(film?.style?.font_file);
 // Refetch when the project changes (a running job may re-time shots or add the narration); the position is kept.
 useEffect(()=>{const c=new AbortController();fetch(`/api/projects/${project.id}/film`,{signal:c.signal}).then(async r=>{const d=await r.json();if(!r.ok)throw new Error(d.detail||'读取失败');setFilm(d)}).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>c.abort()},[project.id,project.revision]);
 const shots=film?.shots||[];
 const index=useMemo(()=>{if(!shots.length)return 0;let lo=0,hi=shots.length-1;while(lo<hi){const mid=(lo+hi+1)>>1;if(shots[mid].start<=t+1e-6)lo=mid;else hi=mid-1}return lo},[shots,t]);
 const row=shots[index], shot=project.shots.find(s=>s.id===row?.id);
 const indexRef=useRef(0);indexRef.current=index;
 const assetOf=(id:string)=>{const s=project.shots.find(x=>x.id===id);return project.assets.find(a=>a.id===s?.selected_asset)};
 const stageRatio=project.aspect==='9:16'?9/16:16/9;
 // The narration is the clock: the full track (only once it is aligned to the shots), or this shot's own audio.
 const full=film?.aligned&&film.audio_path?film.audio_path:null;
 const audioSrc=full||(!film?.audio_path?row?.audio_path:null)||null;
 const now=()=>{
  const a=audio.current;
  if(full&&a&&!a.paused&&!a.ended)return a.currentTime;
  if(!full&&row?.audio_path&&a){if(waiting.current)return clock.current.t;if(!a.paused&&!a.ended)return row.start+a.currentTime}
  return clock.current.t+(playing?(performance.now()-clock.current.at)/1000:0);
 };
 const anchor=(v:number)=>{clock.current={t:v,at:performance.now()}};
 const seek=(next:number)=>{
  if(!film)return;const v=Math.min(film.total,Math.max(0,next));
  time.current=v;anchor(v);setT(v);
  const a=audio.current;
  if(a&&full){if(v<(a.duration||Infinity)-.02)a.currentTime=v;else a.pause()}
  else if(a&&row?.audio_path&&v>=row.start&&v<row.start+row.duration)a.currentTime=Math.min(v-row.start,a.duration||Infinity);   // same shot: no remount
 };
 const stop=()=>{audio.current?.pause();music.current?.pause();videos.current.forEach(v=>v.pause());anchor(time.current);waiting.current=false;setPlaying(false)};
 const play=async()=>{
  if(!film)return;if(playing){stop();return}
  if(time.current>=film.total-.05)seek(0);
  anchor(time.current);setPlaying(true);
  const a=audio.current,inRow=!!row&&time.current>=row.start&&time.current<row.start+row.duration;
  try{
   if(a&&full&&time.current<(a.duration||Infinity)-.02){a.currentTime=time.current;await a.play()}
   else if(a&&!full&&row?.audio_path&&inRow&&time.current-row.start<(a.duration||Infinity)-.02){waiting.current=true;a.currentTime=Math.max(0,time.current-row.start);await a.play()}
   setError('');
  }catch(e){if((e as DOMException)?.name!=='AbortError'){waiting.current=false;setError('配音播放失败，请检查音频文件')}}
  if(music.current&&project.music){music.current.volume=.12;music.current.currentTime=time.current%(music.current.duration||1e9);music.current.play().catch(()=>{})}
 };
 useEffect(()=>{
  if(!playing||!film)return;
  const timer=setInterval(()=>{
   const v=Math.min(now(),film.total);time.current=v;setT(v);
   if(v>=film.total-.02){time.current=film.total;stop();setT(film.total)}
  },50);
  return()=>clearInterval(timer);
 },[playing,film,row?.id,full]);
 // A shot with its own narration: start it from the right point whenever the shot changes; the picture
 // waits for it, so load latency never makes a cut come early.
 useEffect(()=>{
  const a=audio.current;if(!a||!film||full||!row)return;
  if(!row.audio_path){a.pause();waiting.current=false;anchor(time.current);return}
  if(!playing){waiting.current=false;return}
  const offset=Math.max(0,time.current-row.start);
  waiting.current=true;anchor(row.start+offset);
  const start=()=>{if(offset<(a.duration||Infinity)-.02){a.currentTime=offset;a.play().catch(()=>{waiting.current=false})}else waiting.current=false};
  if(a.readyState>=1)start();else a.addEventListener('loadedmetadata',start,{once:true});
  return()=>a.removeEventListener('loadedmetadata',start);
 },[row?.id,row?.audio_path,playing,full]);
 // Keep the current clip in step with the clock; the next clip waits paused at its first frame.
 useEffect(()=>{
  if(!row||!shot)return;
  videos.current.forEach((el,key)=>{
   const id=key.split('#')[0],s=project.shots.find(x=>x.id===id);if(!s)return;
   if(id!==row.id){if(!el.paused)el.pause();if(el.readyState>=1&&Math.abs(el.currentTime-s.clip_start)>.05)el.currentTime=s.clip_start;return}
   const want=s.clip_start+Math.max(0,t-row.start);
   if(el.readyState>=1){
    if(want>=el.duration-.05){if(!el.paused)el.pause();if(Math.abs(el.currentTime-el.duration)>.1)el.currentTime=Math.max(0,el.duration-.04);return}   // clip shorter than the line: hold its last frame, as the export does
    if(Math.abs(el.currentTime-want)>.3)el.currentTime=want;
    if(playing&&el.paused)el.play().catch(()=>{});
    if(!playing&&!el.paused)el.pause();
   }
  });
 },[t,row?.id,playing]);
 useEffect(()=>{
  const key=(e:KeyboardEvent)=>{
   const target=e.target as HTMLElement;
   if(['INPUT','TEXTAREA','SELECT'].includes(target?.tagName)&&(target as HTMLInputElement).type!=='range')return;
   if(e.key==='Escape')close();
   else if(e.key===' '){e.preventDefault();play()}
   else if(e.key==='ArrowRight'&&shots[index+1]){e.preventDefault();seek(shots[index+1].start)}
   else if(e.key==='ArrowLeft'&&row){e.preventDefault();seek(t-row.start>1.5||!shots[index-1]?row.start:shots[index-1].start)}
  };
  window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
 });
 useEffect(()=>()=>{audio.current?.pause();music.current?.pause();videos.current.forEach(v=>v.pause())},[]);
 const onAudioPlaying=()=>{const a=audio.current;if(!a)return;waiting.current=false;anchor(full?a.currentTime:(row?.start||0)+a.currentTime)};
 const onAudioEnded=()=>{
  const a=audio.current;if(!a||!film)return;
  if(full){if(a.currentTime>=film.total-.06){time.current=film.total;stop();setT(film.total)}else anchor(a.currentTime)}   // a timeline longer than the narration plays on silently
  else if(row)anchor(row.start+a.currentTime);
 };
 const local=row?t-row.start:0;
 const line=row?.segments.find(([b,e])=>local>=b&&local<e)?.[2]||(row&&local>=row.duration-.05?row.segments.at(-1)?.[2]:'')||'';
 const missing=project.shots.filter(s=>!project.assets.some(a=>a.id===s.selected_asset)).length;
 // Render the current shot and the next one (preloaded, hidden); keys keep the loaded element across the cut.
 const visible=row?[index,index+1].filter(i=>shots[i]):[];
 return <div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)close()}}><section className="modal film-modal" role="dialog" aria-modal="true" aria-label="全片预览">
  <div className="modal-heading"><div><h2>全片预览</h2><p className="film-sub">{film?`依次播放 ${shots.length} 个分镜 · 共 ${clockText(film.total)} · 与导出相同的裁切、字幕和配音`:'正在读取分镜时间线…'}{missing?` · ${missing} 镜还没有画面`:''}</p></div><button className="icon-button" aria-label="关闭全片预览" onClick={close}><X/></button></div>
  <div className="film-body">
   {film&&!film.aligned&&film.audio_path&&<div className="inline-note warning">{film.alignment_note||'整段配音还没有按真实语音对齐到各个镜头'}：先静音预览画面与字幕顺序；点“自动配素材”或“自动生成整片”完成对齐后，就能带配音预览，导出也需要先对齐。</div>}
   <div className={'asset-preview caption-stage film-stage '+(project.aspect==='9:16'?'portrait':'')}>
    {visible.map(i=>{const r=shots[i],s=project.shots.find(x=>x.id===r.id),a=assetOf(r.id);if(!s)return null;const shown=i===index;
     const f=framing(project,s,a);
     if(!a)return shown?<div key={r.id} className="empty-visual film-empty">此镜还没有画面<br/><small>{s.narration}</small></div>:null;
     if(a.kind==='video'){
      const reg=(key:string)=>(el:HTMLVideoElement|null)=>{if(el)videos.current.set(key,el);else videos.current.delete(key)};
      const ready=(e:React.SyntheticEvent<HTMLVideoElement>)=>{e.currentTarget.currentTime=s.clip_start+(i===indexRef.current?Math.max(0,time.current-r.start):0)};
      return <div key={r.id} className="film-layer" style={{opacity:shown?1:0}}><FramedVideo f={f} preload="auto" src={projectFile(project.id,a.path)} videoRef={reg(r.id)} backdropRef={reg(r.id+'#backdrop')} onLoadedMetadata={ready}/></div>;
     }
     const progress=shown?Math.min(1,Math.max(0,(t-r.start)/r.duration)):0;
     return <div key={r.id} className="film-layer" style={{opacity:shown?1:0}}>{f.fit==='contain'&&<img className="preview-backdrop" src={projectFile(project.id,a.path)} alt=""/>}<img className="moving-photo" style={{objectFit:f.fit,transform:`scale(${(1+progress*.055)/(f.fit==='contain'?1.06:1)})`}} src={projectFile(project.id,a.path)} alt=""/></div>;
    })}
    {row?.overlay_label&&<span className="preview-label">{row.overlay_label}</span>}
    {line&&<span className="preview-caption" style={captionCss(film?.style,family,stageRatio)}>{line}</span>}
    {!film&&!error&&<div className="empty-visual">正在读取…</div>}
   </div>
   {audioSrc&&<audio ref={audio} preload="auto" src={projectFile(project.id,audioSrc)} onPlaying={onAudioPlaying} onEnded={onAudioEnded} onError={()=>{waiting.current=false;anchor(time.current)}}/>}
   {project.music&&<audio ref={music} preload="auto" loop src={projectFile(project.id,project.music.path)}/>}
   <div className="film-controls">
    <button className="primary small" disabled={!film} onClick={play}>{playing?<Pause size={15}/>:<Play size={15}/>}{playing?'暂停':'播放全片'}</button>
    <button className="icon-button" aria-label="上一镜" disabled={!row||index===0} onClick={()=>seek(shots[index-1].start)}><ChevronLeft size={18}/></button>
    <button className="icon-button" aria-label="下一镜" disabled={!shots[index+1]} onClick={()=>seek(shots[index+1].start)}><ChevronRight size={18}/></button>
    <div className="film-track">
     <div className="film-ticks">{film&&shots.map((r,i)=><i key={r.id} className={i===index?'current':''} style={{left:pct(r.start/film.total),width:pct(r.duration/film.total)}} title={`分镜 ${i+1}`}/>)}</div>
     <input aria-label="全片播放位置" type="range" min="0" max={film?.total||0} step="0.04" value={Math.min(t,film?.total||0)} onChange={e=>seek(Number(e.target.value))}/>
    </div>
    <small className="film-time">{clockText(t)} / {clockText(film?.total||0)}</small>
   </div>
   {row&&shot&&<div className="film-now"><b>分镜 {String(index+1).padStart(2,'0')} / {shots.length}</b><span>{shot.narration}</span><button className="text-link" onClick={()=>edit(shot.id)}><Pencil size={12}/>编辑此镜</button></div>}
   <small className="muted-note film-hint">空格 播放/暂停 · ← → 上一镜/下一镜 · Esc 关闭。预览在浏览器里即时拼接，转场可能有轻微卡顿；导出的视频按同一时间线逐帧合成。</small>
   {error&&<p role="alert" className="candidate-warning">{error}</p>}
  </div>
 </section></div>;
}
