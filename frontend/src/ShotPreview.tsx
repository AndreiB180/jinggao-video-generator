import {useEffect, useRef, useState} from 'react';
import type {Asset, Project, Shot} from './types';

type Captions = {segments:[number,number,string][];duration:number;audio_offset:number;audio_path:string|null;overlay_label:string;alignment:string};
const file = (id:string, path:string) => `/api/projects/${id}/files/${path.split('/').map(encodeURIComponent).join('/')}`;

export function ShotPreview({project,shot,asset}:{project:Project;shot:Shot;asset?:Asset}) {
 const [captions,setCaptions]=useState<Captions|null>(null),[at,setAt]=useState(0),[playing,setPlaying]=useState(false),[error,setError]=useState('');
 const video=useRef<HTMLVideoElement>(null),backdrop=useRef<HTMLVideoElement>(null),audio=useRef<HTMLAudioElement>(null),position=useRef(0);
 const stop=()=>{video.current?.pause();backdrop.current?.pause();audio.current?.pause();setPlaying(false)};
 useEffect(()=>{
  const controller=new AbortController();stop();setCaptions(null);setAt(0);position.current=0;setError('');
  fetch(`/api/projects/${project.id}/shots/${shot.id}/captions`,{signal:controller.signal}).then(async r=>{if(!r.ok)throw new Error('字幕读取失败');return r.json()}).then(setCaptions).catch(e=>{if(e.name!=='AbortError')setError(e.message)});
  return()=>controller.abort();
 },[project.id,shot.id,shot.narration,shot.duration,shot.overlay_label,shot.clip_start,asset?.id,project.full_audio?.path,shot.audio?.path]);
 const seek=(time:number)=>{
  const t=Math.min(shot.duration,Math.max(0,time));position.current=t;setAt(t);
  if(video.current)video.current.currentTime=shot.clip_start+t;
  if(backdrop.current)backdrop.current.currentTime=shot.clip_start+t;
  if(audio.current&&captions)audio.current.currentTime=captions.audio_offset+t;
 };
 useEffect(()=>{
  if(!playing)return;
  let previous=performance.now();
  const timer=setInterval(()=>{
   const now=performance.now();
   const t=audio.current&&!audio.current.paused ? audio.current.currentTime-(captions?.audio_offset||0) : video.current ? video.current.currentTime-shot.clip_start : position.current+(now-previous)/1000;
   previous=now;position.current=Math.max(0,t);setAt(position.current);
   if(t>=shot.duration-.025){stop();position.current=shot.duration;setAt(shot.duration)}
  },60);
  return()=>clearInterval(timer);
 },[playing,shot.duration,shot.clip_start,captions]);
 const play=async()=>{
  if(playing){stop();return}setError('');
  try{seek(position.current>=shot.duration-.1?0:position.current);await Promise.all([video.current?.play(),backdrop.current?.play(),audio.current?.play()]);setPlaying(true)}
  catch{stop();setError('预览播放失败，请检查素材或配音文件')}
 };
 const line=captions?.segments.find(([begin,end])=>at>=begin&&at<end)?.[2]||(at>=shot.duration?captions?.segments.at(-1)?.[2]:'');
 const fit=shot.fit==='contain'?'contain':'cover';
 // Match the export's 5.5% zoom and its inset that keeps contain photos inside the frame.
 const photoScale=(1+Math.min(at/shot.duration,1)*.055)/(fit==='contain'?1.06:1);
 return <div className="shot-preview-wrap">
  <div className={'asset-preview caption-stage '+(project.aspect==='9:16'?'portrait':'')}>
   {fit==='contain'&&asset&&(asset.kind==='video'?<video ref={backdrop} className="preview-backdrop" key={'backdrop-'+asset.id} aria-hidden="true" muted playsInline preload="metadata" src={file(project.id,asset.path)} onLoadedMetadata={()=>seek(position.current)}/>:<img className="preview-backdrop" aria-hidden="true" src={file(project.id,asset.path)} alt=""/>)}
   {asset?.kind==='video'?<video ref={video} key={asset.id} style={{objectFit:fit}} muted playsInline preload="metadata" src={file(project.id,asset.path)} onLoadedMetadata={()=>seek(0)} onEnded={stop}/>:asset?<img className="moving-photo" style={{objectFit:fit,transform:`scale(${photoScale})`}} src={file(project.id,asset.path)} alt={asset.title}/>:<div className="empty-visual">画面尚未选定 · 可先预览口播字幕</div>}
   {captions?.overlay_label&&<span className="preview-label">{captions.overlay_label}</span>}
   {line&&<span className="preview-caption">{line}</span>}
  </div>
  {captions?.audio_path&&<audio ref={audio} preload="metadata" src={file(project.id,captions.audio_path)} onLoadedMetadata={()=>{if(audio.current)audio.current.currentTime=captions.audio_offset+position.current}} onEnded={stop}/>}
  <div className="caption-controls"><button className="secondary small" onClick={play}>{playing?'暂停':'播放分镜'}</button><input aria-label="分镜播放位置" type="range" min="0" max={shot.duration} step="0.05" value={Math.min(at,shot.duration)} onChange={e=>seek(Number(e.target.value))}/><small>{Math.min(at,shot.duration).toFixed(1)} / {shot.duration.toFixed(1)} 秒</small></div>
  <small className="muted-note">{captions?.alignment==='local_paraformer'?'字幕已按实际音轨对齐 · 本地处理，不产生模型接口费。':captions?.audio_path?'当前为粗略字幕；导出时先在本地识别音轨并对齐。':'当前未配音，字幕仅作排版预览。'}</small>
  {error&&<p role="alert" className="candidate-warning">{error}</p>}
 </div>;
}
