import {useEffect, useRef, useState} from 'react';
import {Check, ExternalLink, Loader2, Pause, Play, RefreshCw} from 'lucide-react';

export type BaiduVoiceItem = {name:string;description:string;category:string;sample:string};
type VoiceList = {voices:BaiduVoiceItem[];updated:string;voice:string};
type Status = {browser:string|null;open:boolean;logged_in:boolean|null;message:string;voice:string;voices:number;voices_updated:string};

async function call<T>(path:string,method='GET',body?:unknown):Promise<T>{
 const r=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
 const data=await r.json().catch(()=>({}));
 if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'请求没有完成，请重试');
 return data;
}

/** The 一镜 voices last read from the logged-in dedicated browser (kept by the app, no browser needed). */
export function useBaiduVoices(){
 const [list,setList]=useState<VoiceList|null>(null);
 useEffect(()=>{call<VoiceList>('/baidu/voices').then(setList).catch(()=>setList({voices:[],updated:'',voice:''}))},[]);
 return [list,setList] as const;
}

/** A compact select of the 一镜 voices (home screen, batch panel); falls back to the saved name. */
export function BaiduVoiceSelect({value,change}:{value:string;change:(name:string)=>void}){
 const [list]=useBaiduVoices();
 const voices=list?.voices||[];
 if(!voices.length)return <p className="muted-note">百度一镜 · 音色「{value||'未选择'}」。在 设置 → 配音 里登录一镜并刷新音色列表后可在这里换音色。</p>;
 const names=voices.map(v=>v.name);
 return <label>一镜音色<select value={value} onChange={e=>change(e.target.value)}>
  {!names.includes(value)&&<option value={value}>{value||'请选择'}（不在当前列表里）</option>}
  {voices.map(v=><option key={v.name} value={v.name}>{v.name}{noteOf(v)?' · '+noteOf(v):''}</option>)}
 </select></label>;
}

function useSample(){
 const audio=useRef<HTMLAudioElement|null>(null);
 const [playing,setPlaying]=useState(''),[loading,setLoading]=useState(''),[error,setError]=useState('');
 useEffect(()=>()=>{audio.current?.pause()},[]);
 const toggle=async(name:string)=>{
  setError('');
  if(playing===name){audio.current?.pause();setPlaying('');return}
  audio.current?.pause();setLoading(name);
  try{
   const a=new Audio('/api/baidu/voices/sample?name='+encodeURIComponent(name));audio.current=a;
   a.onended=()=>setPlaying('');
   await a.play();setPlaying(name);
  }catch{setError('试听样音没有加载出来（一镜可能没有给这个音色提供样音）')}finally{setLoading('')}
 };
 return {playing,loading,error,toggle};
}

/** Age and use from the 一镜 category (gender is the tab), then the page's own short description. */
const noteOf=(v:BaiduVoiceItem)=>[...(v.category||'').split('、').slice(1),v.description].filter(Boolean).join(' · ');
const groupOf=(v:BaiduVoiceItem)=>{const first=(v.category||'').split(/[、,，\s]/)[0];return first==='男'?'男声':first==='女'?'女声':'其他'};

/** Settings → 配音 when 百度一镜 is chosen: dedicated browser and login, then pick one of the 一镜 voices. */
export function BaiduSettings({values,change}:{values:Record<string,any>;change:(key:string,value:string)=>void}){
 const [status,setStatus]=useState<Status|null>(null),[note,setNote]=useState(''),[busy,setBusy]=useState('');
 const [list,setList]=useBaiduVoices();
 const [group,setGroup]=useState('全部'),[filter,setFilter]=useState('');
 const sample=useSample();
 const refreshed=useRef(false);
 const loadStatus=()=>call<Status>('/baidu/status').then(setStatus).catch(e=>setNote((e as Error).message));
 useEffect(()=>{loadStatus()},[]);
 // While the dedicated browser waits for the QR scan, follow the login; then read the voices once.
 useEffect(()=>{
  if(!status?.open||status.logged_in)return;
  const t=setInterval(loadStatus,3000);return()=>clearInterval(t);
 },[status?.open,status?.logged_in]);
 useEffect(()=>{if(status?.logged_in&&list&&!list.voices.length&&!refreshed.current){refreshed.current=true;refresh()}},[status?.logged_in,list]);
 const open=async()=>{setBusy('open');setNote('');try{const r=await call<{message:string}>('/baidu/browser','POST');setNote(r.message);await loadStatus()}catch(e){setNote((e as Error).message)}finally{setBusy('')}};
 const refresh=async()=>{setBusy('refresh');setNote('');try{const r=await call<VoiceList>('/baidu/voices/refresh','POST');setList(r);setNote(`已读取 ${r.voices.length} 个一镜音色`)}catch(e){setNote((e as Error).message)}finally{setBusy('')}};
 const chooseBrowser=(value:string)=>{change('baidu_browser',value);call('/settings','PATCH',{baidu_browser:value}).then(loadStatus).catch(e=>setNote((e as Error).message))};
 const voice=(values.baidu_voice_name||'').trim();
 const voices=list?.voices||[];
 const groups=['全部',...new Set(voices.map(groupOf))];
 const shown=voices.filter(v=>(group==='全部'||groupOf(v)===group)&&(!filter||v.name.includes(filter)||(v.description||'').includes(filter)));
 const state=!status?'':status.logged_in?'已登录':status.open?'未登录':'未打开';
 return <div className="baidu-settings">
  <p>需要先点下面的按钮，在镜稿的一镜专用浏览器里登录百度账号（用百度 App 扫码），登录一次即可，之后会一直保持登录。生成配音时镜稿在这个浏览器里打开一镜音色库、按你选的音色下载口播（网页显示「限时免费」时才下载），不影响你自己平时用的浏览器。单次最多 5000 字。</p>
  <div className="baidu-login">
   <button className="secondary small" disabled={!!busy} onClick={open}>{busy==='open'?<Loader2 className="spin" size={15}/>:<ExternalLink size={15}/>}{busy==='open'?'正在打开……':'打开一镜专用浏览器并登录'}</button>
   {status&&<span className={'baidu-state '+(status.logged_in?'ok':'')}><i className={status.logged_in?'green-dot':'pending-dot'}/>{state}{status.browser?` · ${status.browser}`:''}</span>}
  </div>
  {(note||status?.message)&&<p className="muted-note">{note||status?.message}</p>}
  <div className="baidu-voice-head">
   <strong>一镜音色</strong><span>当前：{voice||'未选择'}</span>
   <button className="secondary small" disabled={!!busy} onClick={refresh} title="在已登录的专用浏览器里重新读取一镜公共音色">{busy==='refresh'?<Loader2 className="spin" size={14}/>:<RefreshCw size={14}/>}刷新音色列表</button>
  </div>
  {voices.length>0?<div className="voice-picker">
   <div className="baidu-filter"><div className="segmented small" role="tablist">{groups.map(g=><button key={g} role="tab" aria-selected={g===group} className={g===group?'on':''} onClick={()=>setGroup(g)}>{g}</button>)}</div>
    <input value={filter} placeholder="搜索音色名称" onChange={e=>setFilter(e.target.value)}/></div>
   {voice&&!voices.some(v=>v.name===voice)&&<div className="inline-note warning">当前音色「{voice}」不在最新的一镜音色列表里（可能已下架或改名），请重新选一个。</div>}
   <div className="voice-grid">{shown.map(v=><div key={v.name} className={'voice-card'+(v.name===voice?' chosen':'')}>
    <button className="voice-choose" onClick={()=>change('baidu_voice_name',v.name)} aria-pressed={v.name===voice}>
     <strong>{v.name}{v.name===voice&&<Check size={14}/>}</strong><small title={noteOf(v)}>{noteOf(v)||'一镜公共音色'}</small>
    </button>
    {v.sample&&<button className="voice-play" aria-label={'试听 '+v.name} onClick={()=>sample.toggle(v.name)}>{sample.loading===v.name?<Loader2 className="spin" size={15}/>:sample.playing===v.name?<Pause size={15}/>:<Play size={15}/>}</button>}
   </div>)}</div>
   {!shown.length&&<p className="muted-note">没有符合条件的音色。</p>}
   {sample.error&&<p className="field-error">{sample.error}</p>}
   <small className="muted-note">共 {voices.length} 个音色{list?.updated?`，${list.updated.replace('T',' ')} 读取`:''}。选好后点「保存设置」。</small>
  </div>:<>
   <p className="muted-note">还没有读取一镜音色列表：登录后点「刷新音色列表」。也可以直接填写一镜公共音色的完整名称：</p>
   <label>一镜音色名称<input value={values.baidu_voice_name||''} onChange={e=>change('baidu_voice_name',e.target.value)}/></label>
  </>}
  <label>一镜专用浏览器<select value={values.baidu_browser||'auto'} onChange={e=>chooseBrowser(e.target.value)}>
   <option value="auto">自动（先用 Edge，没有就用 Chrome）</option><option value="edge">Microsoft Edge</option><option value="chrome">Google Chrome</option>
  </select></label>
 </div>;
}
