import {useEffect, useRef, useState} from 'react';
import {Check, Loader2, QrCode} from 'lucide-react';

type Status = {logged_in:boolean;verified?:boolean;name?:string;vip?:boolean};

export function BilibiliLogin() {
  const [status,setStatus]=useState<Status|null>(null);
  const [image,setImage]=useState('');
  const [message,setMessage]=useState('');
  const [busy,setBusy]=useState(false);
  const timer=useRef<number|undefined>(undefined);
  const refresh=async()=>{try{setStatus(await (await fetch('/api/bilibili/status')).json())}catch{setStatus(null)}};
  useEffect(()=>{refresh();return()=>window.clearInterval(timer.current)},[]);
  const start=async()=>{
    setBusy(true);setMessage('');window.clearInterval(timer.current);
    try{
      const r=await (await fetch('/api/bilibili/qr',{method:'POST'})).json();
      setImage(r.image);setMessage('请用哔哩哔哩手机客户端扫码并确认登录');
      timer.current=window.setInterval(async()=>{
        const s=await (await fetch('/api/bilibili/qr/'+r.key)).json();
        if(s.status==='scanned')setMessage('已扫码，请在手机上确认');
        if(s.status==='success'){window.clearInterval(timer.current);setImage('');setMessage('登录成功');refresh()}
        if(s.status==='expired'){window.clearInterval(timer.current);setImage('');setMessage('二维码已过期，请重新获取')}
      },2000);
    }catch(e){setMessage((e as Error).message)}finally{setBusy(false)}
  };
  return <div className="stock-key">
    <label>哔哩哔哩素材（720p/1080p）</label>
    <p>{status?.logged_in?<><Check size={14}/> 已登录{status.name?`：${status.name}`:''}。</>:'未登录时 B 站多数视频只提供 480p，会被自动跳过；扫码登录一次即可获取 720p/1080p 实拍素材。登录信息只保存在本机。'}</p>
    <button className="secondary small full" disabled={busy} onClick={start}>{busy?<Loader2 className="spin" size={15}/>:<QrCode size={15}/>}{status?.logged_in?'重新扫码登录':'扫码登录 B 站'}</button>
    {image&&<img src={image} alt="哔哩哔哩登录二维码" style={{width:200,height:200,display:'block',margin:'10px auto',background:'#fff'}}/>}
    {message&&<p>{message}</p>}
  </div>;
}
