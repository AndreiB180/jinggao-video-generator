import {useCallback, useEffect, useState} from 'react';
import {Check, Copy, FolderOpen, FolderSearch, Play} from 'lucide-react';

async function post<T>(path:string,body?:unknown):Promise<T>{
 const r=await fetch('/api'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body??{})});
 const d=await r.json();
 if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:'操作没有成功');
 return d;
}
/** The file name of a Windows or macOS path. */
export const fileName=(path:string)=>path.split(/[\\/]/).filter(Boolean).pop()||path;

/** Where finished videos are saved, with buttons to open the folder or pick another one. */
export function useOutputFolder(){
 const [folder,setFolder]=useState(''),[note,setNote]=useState('');
 const load=useCallback(()=>fetch('/api/output-folder').then(r=>r.json()).then(d=>setFolder(d.path||'')).catch(()=>{}),[]);
 useEffect(()=>{load()},[load]);
 const open=()=>post<{path:string}>('/output-folder/open').then(d=>{setFolder(d.path);setNote('已在文件管理器中打开')}).catch(e=>setNote(e.message));
 const choose=()=>{setNote('请在弹出的窗口里选择文件夹（可能在浏览器后面）');
  return post<{path:string;changed:boolean}>('/output-folder/choose').then(d=>{setFolder(d.path);setNote(d.changed?'之后做好的视频保存到这里':'')}).catch(e=>setNote(e.message))};
 return {folder,note,open,choose,reload:load};
}

export function FolderBar({label='做好的视频自动保存在'}:{label?:string}){
 const {folder,note,open,choose}=useOutputFolder();
 if(!folder)return null;
 return <div className="folder-bar">
  <FolderOpen size={16}/>
  <div><span>{label}</span><code title={folder}>{folder}</code>{note&&<small>{note}</small>}</div>
  <button className="secondary small" onClick={open}><FolderOpen size={14}/>打开文件夹</button>
  <button className="text-link" onClick={choose}>更改位置</button>
 </div>;
}

/** Play a finished video in the computer's own player, show it selected in its folder, or copy its path. */
export function VideoFileActions({pid,path,exportId,compact}:{pid:string;path?:string;exportId?:string;compact?:boolean}){
 const [note,setNote]=useState(''),[copied,setCopied]=useState(false);
 const act=(what:'play'|'reveal')=>post<{path:string}>(`/projects/${pid}/${what}`,{export_id:exportId}).then(()=>setNote(what==='play'?'已用播放器打开':'已在文件夹中选中这个视频')).catch(e=>setNote(e.message));
 const copy=()=>{if(!path)return;navigator.clipboard?.writeText(path).then(()=>{setCopied(true);setTimeout(()=>setCopied(false),1500)}).catch(()=>setNote('复制没有成功，可以直接选中路径文字复制'))};
 if(compact)return <span className="file-actions compact" onClick={e=>e.stopPropagation()}>
  <button className="icon-button" title="用播放器打开" aria-label="用播放器打开" onClick={()=>act('play')}><Play size={14}/></button>
  <button className="icon-button" title="在文件夹中显示" aria-label="在文件夹中显示" onClick={()=>act('reveal')}><FolderSearch size={14}/></button>
  {note&&<small>{note}</small>}
 </span>;
 return <div className="file-actions">
  <button className="secondary small" onClick={()=>act('play')}><Play size={14}/>用播放器打开</button>
  <button className="secondary small" onClick={()=>act('reveal')}><FolderSearch size={14}/>在文件夹中显示</button>
  {path&&<button className="secondary small" onClick={copy}>{copied?<Check size={14}/>:<Copy size={14}/>}{copied?'已复制':'复制路径'}</button>}
  {note&&<small>{note}</small>}
 </div>;
}
