import {useEffect, useState} from 'react';
import {Check, ClipboardCopy, Cpu, Loader2, RefreshCw} from 'lucide-react';

export type SystemInfo = {ffmpeg?:boolean;nvenc?:boolean;full?:boolean;nvidia?:boolean;gpu?:string;driver?:string;
 reason?:string;advice?:string;summary?:string;cores?:number;cpu_slots?:number;gpu_slots?:number;sessions?:number;version?:string};

/** Whether the NVIDIA card is really used, why not, and a report to send for support. */
export function GpuStatus({onChange}:{onChange?:(s:SystemInfo)=>void}){
 const [info,setInfo]=useState<SystemInfo|null>(null),[busy,setBusy]=useState(false),[note,setNote]=useState('');
 const show=(s:SystemInfo)=>{setInfo(s);onChange?.(s)};
 useEffect(()=>{fetch('/api/system').then(r=>r.json()).then(show).catch(()=>{})},[]);
 const recheck=async()=>{setBusy(true);setNote('');try{const r=await fetch('/api/system/gpu-check',{method:'POST'});show(await r.json());setNote('已重新检测')}catch{setNote('检测没有完成，请稍后再试')}finally{setBusy(false)}};
 const copy=async()=>{try{const text=await (await fetch('/api/system/report')).text();await navigator.clipboard.writeText(text);setNote('诊断信息已复制，可以直接粘贴到微信发给我们')}catch{setNote('复制没有成功：可以在浏览器打开 /api/system/report 手动复制')}};
 if(!info)return null;
 const ok=info.nvenc, state=info.full?'显卡加速已开启':info.nvenc?'显卡编码已开启（解码仍用 CPU）':info.nvidia?'显卡加速没有开启':'没有检测到 NVIDIA 显卡';
 return <div className={'gpu-box '+(ok?'ok':'off')}>
  <strong>{ok?<Check size={14}/>:<Cpu size={14}/>} {state}</strong>
  {info.nvidia&&<p>{info.gpu} · 驱动 {info.driver||'未知'}{ok?` · 同时 ${info.gpu_slots||info.sessions} 路显卡编码`:''}</p>}
  {info.full&&<p>候选片段的解码、缩放、编码和最终合成都交给显卡，CPU 只做去水印分析和字幕。</p>}
  {!ok&&info.advice&&<p>{info.advice}</p>}
  {info.nvidia&&!info.full&&info.reason&&<pre>{info.reason}</pre>}
  <p>CPU：{info.cores} 个逻辑核心，最多同时 {info.cpu_slots} 个 CPU 转码；制作进程以「低于正常」优先级运行，前台软件优先，电脑不会被占得发卡。</p>
  <div className="file-actions">
   <button className="secondary small" disabled={busy} onClick={recheck}>{busy?<Loader2 className="spin" size={14}/>:<RefreshCw size={14}/>}重新检测</button>
   <button className="secondary small" onClick={copy}><ClipboardCopy size={14}/>复制诊断信息</button>
   {note&&<small>{note}</small>}
  </div>
 </div>;
}
