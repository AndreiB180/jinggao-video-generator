import {useEffect, useState} from 'react';
import {Check, ClipboardCopy, Cpu, Headset, Loader2, RefreshCw} from 'lucide-react';

export type AssistState = {available?:boolean;state?:'off'|'connecting'|'waiting'|'on'|'failed';port?:number|null;message?:string;key?:string};

/** A line in the sidebar while 远程协助 is on, so the customer always sees it. */
export function AssistBadge(){
 const [s,setS]=useState<AssistState|null>(null);
 useEffect(()=>{
  const load=()=>fetch('/api/assist').then(r=>r.json()).then(setS).catch(()=>{});
  load();const timer=setInterval(load,10000);return()=>clearInterval(timer);
 },[]);
 return s?.state==='on'?<small className="assist-on">远程协助中 · 协助码 {s.port}（设置里可结束）</small>:null;
}

/** 远程协助: the customer turns it on, tells us the code, and we can open this 镜稿 page from our side. */
export function AssistBox(){
 const [s,setS]=useState<AssistState|null>(null),[busy,setBusy]=useState(false),[copied,setCopied]=useState(false);
 useEffect(()=>{
  const load=()=>fetch('/api/assist').then(r=>r.json()).then(setS).catch(()=>{});
  load();const timer=setInterval(load,3000);return()=>clearInterval(timer);
 },[]);
 if(!s?.available)return null;
 const act=async(path:string)=>{setBusy(true);try{setS(await (await fetch(path,{method:'POST'})).json())}catch{/* next poll shows the state */}finally{setBusy(false)}};
 const copyKey=async()=>{try{await navigator.clipboard.writeText(s.key||'');setCopied(true)}catch{/* the text can be selected by hand */}};
 const running=s.state==='on'||s.state==='connecting'||s.state==='waiting';
 return <div className={'assist-box '+(s.state||'off')}>
  <strong><Headset size={14}/> 远程协助{s.state==='on'?' · 已连接':''}</strong>
  <p>开启后，技术人员可以远程打开这台电脑上的镜稿页面：看每条视频卡在哪一步，在这台电脑的网络上做检测。只能看到镜稿，看不到电脑里的其他文件和软件。用完点「结束」；关掉镜稿或开满 12 小时也会自动结束。</p>
  {s.state==='on'&&<p className="assist-code">协助码 <b>{s.port}</b>　把这个数字发给技术人员</p>}
  {s.state==='connecting'&&<p>{s.message||'正在连接…'}</p>}
  {s.state==='waiting'&&<>
   <p>第一次使用：请把下面这段「电脑识别码」复制发给技术人员。对方加好后这里会自动变成「已连接」，以后再开就不用重复这一步。</p>
   <pre className="assist-key">{s.key}</pre>
   <div className="file-actions"><button className="secondary small" onClick={copyKey}><ClipboardCopy size={14}/>{copied?'已复制':'复制识别码'}</button></div>
  </>}
  {s.state==='failed'&&<p className="assist-error">{s.message}</p>}
  <div className="file-actions">
   {running?<button className="secondary small" disabled={busy} onClick={()=>act('/api/assist/stop')}>结束远程协助</button>
    :<button className="primary small" disabled={busy} onClick={()=>act('/api/assist/start')}>{busy?<Loader2 className="spin" size={14}/>:<Headset size={14}/>}开始远程协助</button>}
  </div>
 </div>;
}

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
