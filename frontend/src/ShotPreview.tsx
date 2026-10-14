import {useEffect, useRef, useState} from 'react';
import type {Asset, FrameCrop, Project, Shot} from './types';

export type CaptionStyle = {font:string;font_file:string;colour:string;size:number;outline:number;margin:number};
type Captions = {segments:[number,number,string][];duration:number;audio_offset:number;audio_path:string|null;overlay_label:string;alignment:string;style?:CaptionStyle};
const file = (id:string, path:string) => `/api/projects/${id}/files/${path.split('/').map(encodeURIComponent).join('/')}`;

const loadedFonts = new Set<string>();
/** The burned-in subtitle font, loaded from the local service so the preview matches the export. */
export function useSubtitleFont(fontFile?:string) {
 const [family,setFamily]=useState('');
 useEffect(()=>{
  if(!fontFile){setFamily('');return}
  const name='jg-'+fontFile.replace(/[^A-Za-z0-9]/g,'');
  if(loadedFonts.has(name)){setFamily(name);return}
  let alive=true;
  new FontFace(name,`url(/api/fonts/${encodeURIComponent(fontFile)})`).load().then(face=>{document.fonts.add(face);loadedFonts.add(name);if(alive)setFamily(name)}).catch(()=>{if(alive)setFamily('')});
  return()=>{alive=false};
 },[fontFile]);
 return family;
}

/** Caption look in container units (cqw) of a stage with the given width/height ratio. */
export function captionCss(style:CaptionStyle|undefined, family:string, stageRatio:number):React.CSSProperties {
 if(!style)return {};
 const unit=(fraction:number)=>`${(fraction*100/stageRatio).toFixed(3)}cqw`;
 return {fontFamily:`${family?`"${family}",`:''}"Noto Sans SC",sans-serif`,fontSize:unit(style.size),color:style.colour,fontWeight:family?400:800,
  WebkitTextStroke:`${unit(style.outline*2)} #000`,paintOrder:'stroke fill',textShadow:`0 ${unit(style.outline*.6)} ${unit(style.outline)} #000a`,
  bottom:`${(style.margin*100).toFixed(2)}%`,lineHeight:1.25,whiteSpace:'pre'};
}

export const pct=(v:number)=>`${(v*100).toFixed(3)}%`;
export const projectFile=file;

export type Box={left:number;top:number;w:number;h:number};
/** The asset is the measured file or a privacy-redacted copy of it (media._derived_from). */
function derivedFrom(project:Project, asset:Asset, id:string) {
 const seen=new Set<string>();let current:Asset|undefined=asset;
 while(current&&!seen.has(current.id)){if(current.id===id)return true;seen.add(current.id);const origin:string|undefined=(current as any).privacy_redaction?.original_asset_id;current=origin?project.assets.find(a=>a.id===origin):undefined}
 return false;
}
export const boxStyle=(b:Box):React.CSSProperties=>({position:'absolute',left:pct(b.left),top:pct(b.top),width:pct(b.w),height:pct(b.h)});

/** Where an image of the given width/height ratio lands on the stage (fractions of the stage). */
function displayBox(ratio:number, stageRatio:number, fit:'contain'|'cover'):Box {
 const [w,h]=(fit==='cover')===(ratio>stageRatio)?[ratio/stageRatio,1]:[1,stageRatio/ratio];
 return {left:(1-w)/2,top:(1-h)/2,w,h};
}

/** How the export frames this shot's asset (media.render): it first cuts the measured clean rectangle
 * (express frame_crop, or the bottom subtitle band crop_bottom) from the file it was measured on, then
 * fills (cover) or fits (contain, over a blurred copy) the stage. */
export function framing(project:Project, shot:Shot, asset:Asset|undefined, view:'clean'|'original'='clean') {
 const fit:'contain'|'cover'=shot.fit==='contain'?'contain':'cover';
 const stageRatio=project.aspect==='9:16'?9/16:16/9;
 const srcRatio=asset?.width&&asset?.height?asset.width/asset.height:stageRatio;
 const measured=asset?.kind==='video'&&(!shot.crop_asset||derivedFrom(project,asset,shot.crop_asset));
 const fc=measured&&shot.frame_crop?shot.frame_crop:undefined;
 const bottom=measured&&!fc&&(shot.crop_bottom||0)>0&&(shot.crop_bottom||0)<.3?shot.crop_bottom!:0;
 const crop:number[]|null=view==='clean'?(fc?fc.crop:bottom?[0,0,1,1-bottom]:null):null;
 const videoStyle:React.CSSProperties={objectFit:fc&&view==='original'?'contain':fit};
 if(!crop)return {fit,stageRatio,srcRatio,fc,crop,window:null,inner:null,patches:[] as Box[],backdropWindow:null,videoStyle};
 const [x,y,cw,ch]=crop, ratio=srcRatio*cw/ch;
 return {fit,stageRatio,srcRatio,fc,crop,videoStyle,
  window:displayBox(ratio,stageRatio,fit),                       // the cut rectangle on the stage
  inner:{left:-x/cw,top:-y/ch,w:1/cw,h:1/ch} as Box,            // the whole file, relative to that window
  patches:(fc?.residual||[]).map(r=>({left:(r.box[0]-x)/cw,top:(r.box[1]-y)/ch,w:(r.box[2]-r.box[0])/cw,h:(r.box[3]-r.box[1])/ch})),
  backdropWindow:fit==='contain'?displayBox(ratio,stageRatio,'cover'):null};
}
export type Framing=ReturnType<typeof framing>;

/** A video framed as the export cuts it, with the blurred fill behind contain shots. */
export function FramedVideo({f,src,videoRef,backdropRef,onLoadedMetadata,onEnded,preload='metadata'}:{f:Framing;src:string;videoRef?:React.Ref<HTMLVideoElement>;backdropRef?:React.Ref<HTMLVideoElement>;onLoadedMetadata?:React.ReactEventHandler<HTMLVideoElement>;onEnded?:()=>void;preload?:string}) {
 const inner=f.inner?{...boxStyle(f.inner),objectFit:'fill' as const,maxWidth:'none'}:undefined;
 return <>
  {f.fit==='contain'&&(f.backdropWindow?<div className="crop-window preview-backdrop-window" aria-hidden="true" style={boxStyle(f.backdropWindow)}><video ref={backdropRef} muted playsInline preload={preload} src={src} style={inner} onLoadedMetadata={onLoadedMetadata}/></div>
   :<video ref={backdropRef} className="preview-backdrop" aria-hidden="true" muted playsInline preload={preload} src={src} onLoadedMetadata={onLoadedMetadata}/>)}
  {f.window?<div className="crop-window" style={boxStyle(f.window)}><video ref={videoRef} muted playsInline preload={preload} src={src} style={inner} onLoadedMetadata={onLoadedMetadata} onEnded={onEnded}/>{f.patches.map((b,i)=><span key={i} className="blur-patch" style={boxStyle(b)}/>)}</div>
   :<video ref={videoRef} muted playsInline preload={preload} src={src} style={f.videoStyle} onLoadedMetadata={onLoadedMetadata} onEnded={onEnded}/>}
 </>;
}

const markNames:Record<string,string>={logo:'台标/水印',subtitle:'原片字幕',center:'画面中部水印'};

function cleaningSummary(fc:FrameCrop|undefined, flags:string[]):string {
 if(!fc)return '';
 const count=(kind:string)=>(fc.boxes?fc.boxes.filter(b=>b.kind===kind).length:fc.overlays.filter(k=>k===kind).length);
 const parts=[];
 if(count('logo'))parts.push(`裁掉台标/水印 ${count('logo')} 处`);
 if(count('subtitle'))parts.push(`挡掉原片字幕 ${count('subtitle')} 行`);
 if(fc.residual.length)parts.push(`模糊 ${fc.residual.length} 处`);
 if(fc.zoom>1.01)parts.push(`画面放大 ${fc.zoom.toFixed(2)} 倍`);
 const text=parts.length?'自动去水印：'+parts.join(' · '):'自动检查：未发现台标、水印或原片字幕';
 const notes=[flags.includes('wm')?'抽检仍可能残留水印，已优先换片未果':'',flags.includes('still')?'近似静态图片':''].filter(Boolean);
 return text+(notes.length?'（'+notes.join('；')+'）':'');
}

export function ShotPreview({project,shot,asset}:{project:Project;shot:Shot;asset?:Asset}) {
 const [captions,setCaptions]=useState<Captions|null>(null),[at,setAt]=useState(0),[playing,setPlaying]=useState(false),[error,setError]=useState('');
 const [view,setView]=useState<'clean'|'original'>('clean');
 const video=useRef<HTMLVideoElement>(null),backdrop=useRef<HTMLVideoElement>(null),audio=useRef<HTMLAudioElement>(null),position=useRef(0);
 const family=useSubtitleFont(captions?.style?.font_file);
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
 const hasMarks=!!framing(project,shot,asset,'clean').fc, shownView=hasMarks?view:'clean';
 const framed=framing(project,shot,asset,shownView), {fit,stageRatio,srcRatio,fc}=framed;
 // Match the export's 5.5% zoom and its inset that keeps contain photos inside the frame.
 const photoScale=(1+Math.min(at/shot.duration,1)*.055)/(fit==='contain'?1.06:1);
 const flags:string[]=(shot.verification as any)?.express?.flags||[];
 const summary=cleaningSummary(fc,flags);
 // Where the detected marks sit on the uncropped frame (contain layout).
 const shown=srcRatio>stageRatio?{x:0,y:(1-stageRatio/srcRatio)/2,w:1,h:stageRatio/srcRatio}:{x:(1-srcRatio/stageRatio)/2,y:0,w:srcRatio/stageRatio,h:1};
 const onFrame=(b:number[])=>({left:pct(shown.x+b[0]*shown.w),top:pct(shown.y+b[1]*shown.h),width:pct((b[2]-b[0])*shown.w),height:pct((b[3]-b[1])*shown.h)});
 return <div className="shot-preview-wrap">
  <div className={'asset-preview caption-stage '+(project.aspect==='9:16'?'portrait':'')}>
   {asset?.kind==='video'?<FramedVideo key={asset.id+shownView} f={framed} src={file(project.id,asset.path)} videoRef={video} backdropRef={backdrop} onLoadedMetadata={()=>seek(position.current)} onEnded={stop}/>
    :<>{fit==='contain'&&asset&&<img className="preview-backdrop" aria-hidden="true" src={file(project.id,asset.path)} alt=""/>}{asset?<img className="moving-photo" style={{objectFit:fit,transform:`scale(${photoScale})`}} src={file(project.id,asset.path)} alt={asset.title}/>:<div className="empty-visual">画面尚未选定 · 可先预览口播字幕</div>}</>}
   {fc&&shownView==='original'&&<>
    {(fc.boxes||[]).map((b,i)=><span key={i} className={'mark-box '+b.kind} style={onFrame(b.box)}><i>{markNames[b.kind]||b.kind}</i></span>)}
    <span className="mark-box keep" style={onFrame([fc.crop[0],fc.crop[1],fc.crop[0]+fc.crop[2],fc.crop[1]+fc.crop[3]])}><i>成片保留区域</i></span>
   </>}
   {captions?.overlay_label&&<span className="preview-label">{captions.overlay_label}</span>}
   {line&&shownView==='clean'&&<span className="preview-caption" style={captionCss(captions?.style,family,stageRatio)}>{line}</span>}
  </div>
  {captions?.audio_path&&<audio ref={audio} preload="metadata" src={file(project.id,captions.audio_path)} onLoadedMetadata={()=>{if(audio.current)audio.current.currentTime=captions.audio_offset+position.current}} onEnded={stop}/>}
  <div className="caption-controls"><button className="secondary small" onClick={play}>{playing?'暂停':'播放分镜'}</button><input aria-label="分镜播放位置" type="range" min="0" max={shot.duration} step="0.05" value={Math.min(at,shot.duration)} onChange={e=>seek(Number(e.target.value))}/><small>{Math.min(at,shot.duration).toFixed(1)} / {shot.duration.toFixed(1)} 秒</small></div>
  {fc&&<div className={'clean-note '+(flags.includes('wm')?'warn':'')}><span>{summary}</span><div className="view-switch" role="group" aria-label="画面视图"><button className={view==='clean'?'active':''} onClick={()=>{stop();setView('clean')}}>去水印后</button><button className={view==='original'?'active':''} onClick={()=>{stop();setView('original')}}>原画 · 标出检测</button></div></div>}
  <small className="muted-note">{captions?.alignment==='local_paraformer'?'字幕已按实际音轨对齐 · 本地处理，不产生模型接口费。':captions?.audio_path?'当前为粗略字幕；导出时先在本地识别音轨并对齐。':'当前未配音，字幕仅作排版预览。'}</small>
  {error&&<p role="alert" className="candidate-warning">{error}</p>}
 </div>;
}
