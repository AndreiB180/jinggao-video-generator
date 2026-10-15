import {useEffect, useRef, useState} from 'react';
import {Check, Loader2, Pause, Play} from 'lucide-react';

export type Voice = {id:string;name:string;note:string;group:string};
export type VoiceList = {provider:string;voice:string;voice_key:string;speed:string;speeds:string[];voices:Voice[];
 price_per_1000_chars:number;key_set:boolean};

const cached:Record<string,Promise<VoiceList>>={};
/** The voices of one provider (the free Edge voices or 千问配音). */
export function useVoices(provider:string) {
 const [list,setList]=useState<VoiceList|null>(null);
 useEffect(()=>{
  setList(null);
  cached[provider]=cached[provider]||fetch('/api/tts/voices?provider='+encodeURIComponent(provider)).then(r=>{if(!r.ok)throw new Error('voices');return r.json()});
  cached[provider].then(setList).catch(()=>{delete cached[provider]});
 },[provider]);
 return list;
}

/** One shared player: starting a sample stops the one before. */
function useSample() {
 const audio=useRef<HTMLAudioElement|null>(null);
 const [playing,setPlaying]=useState(''),[loading,setLoading]=useState(''),[error,setError]=useState('');
 useEffect(()=>()=>{audio.current?.pause()},[]);
 const toggle=async(id:string)=>{
  setError('');
  if(playing===id){audio.current?.pause();setPlaying('');return}
  audio.current?.pause();setLoading(id);
  try{
   const r=await fetch('/api/tts/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voice:id})});
   const data=await r.json();if(!r.ok)throw new Error(data.detail||'试听失败');
   const a=new Audio(data.url);audio.current=a;a.onended=()=>setPlaying('');
   await a.play();setPlaying(id);
  }catch(e){setError((e as Error).message)}finally{setLoading('')}
 };
 return {playing,loading,error,toggle};
}

export function voiceLabel(list:VoiceList|null,id:string){const v=list?.voices.find(v=>v.id===id);return v?v.name:id}

/** Voice cards grouped by use, each with a sample; plus the reading speed. */
export function VoicePicker({provider,values,change,compact=false}:{provider:string;values:Record<string,any>;change:(key:string,value:string)=>void;compact?:boolean}) {
 const list=useVoices(provider), sample=useSample();
 const [group,setGroup]=useState('解说');
 if(!list) return <p className="muted-note"><Loader2 className="spin" size={14}/> 正在读取音色…</p>;
 if(!list.voices.length) return null;
 const voice=values[list.voice_key]||list.voice, speed=values.tts_speed||list.speed;
 const groups=[...new Set(list.voices.map(v=>v.group))];
 const shown=list.voices.filter(v=>v.group===group);
 return <div className={'voice-picker'+(compact?' compact':'')}>
  <div className="segmented small" role="tablist">{groups.map(g=><button key={g} role="tab" aria-selected={g===group} className={g===group?'on':''} onClick={()=>setGroup(g)}>{g}</button>)}</div>
  <div className="voice-grid">{shown.map(v=><div key={v.id} className={'voice-card'+(v.id===voice?' chosen':'')}>
   <button className="voice-choose" onClick={()=>change(list.voice_key,v.id)} aria-pressed={v.id===voice}>
    <strong>{v.name}{v.id===voice&&<Check size={14}/>}</strong><small>{v.note}</small>
   </button>
   <button className="voice-play" aria-label={'试听 '+v.name} onClick={()=>sample.toggle(v.id)}>{sample.loading===v.id?<Loader2 className="spin" size={15}/>:sample.playing===v.id?<Pause size={15}/>:<Play size={15}/>}</button>
  </div>)}</div>
  {sample.error&&<p className="field-error">{sample.error}</p>}
  <div className="speed-row"><span>语速</span><div className="segmented small">{list.speeds.map(s=><button key={s} className={s===speed?'on':''} onClick={()=>change('tts_speed',s)}>{s==='1.0'?'正常':s+'×'}</button>)}</div></div>
 </div>;
}
