import {useState} from 'react';
import {ExternalLink, Loader2} from 'lucide-react';

type CheckResult = {status:string;reason:string;checked:boolean;cache_hit:boolean;result_count:number};
const providers = [
  {id:'pixabay',name:'Pixabay',url:'https://pixabay.com/api/docs/',hint:'这里接入的是视频搜索，可获取 MP4 素材。登录官方 API 页面后，在 key 参数旁复制你的 Key。免费 API；搜索结果缓存 24 小时。'},
  {id:'pexels',name:'Pexels',url:'https://www.pexels.com/api/',hint:'目前官网暂停发放新 Key（2026-10-10 核实）；已有 Key 可以配置并验证。'},
];

export function StockKeys({values,change}:{values:Record<string,any>;change:(key:string,value:string)=>void}) {
  const [busy,setBusy]=useState('');
  const [messages,setMessages]=useState<Record<string,string>>({});
  const verify=async(provider:string)=>{
    setBusy(provider);
    setMessages(current=>({...current,[provider]:''}));
    try {
      const response=await fetch('/api/settings/stock/check',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({provider,api_key:values[provider+'_api_key']||''}),
      });
      const result=await response.json();
      if(!response.ok) throw new Error(typeof result.detail==='string'?result.detail:'验证请求失败');
      const checked=result as CheckResult;
      const success=checked.checked&&checked.status==='configured';
      const message=success
        ? `${checked.cache_hit?'已复用 24 小时内成功查询的缓存，未重新联网验证':'官方接口查询成功'}，返回 ${checked.result_count} 条视频。${values[provider+'_api_key']?'点击「保存设置」后生效。':''}`
        : checked.reason;
      setMessages(current=>({...current,[provider]:message}));
    } catch(error) {
      setMessages(current=>({...current,[provider]:(error as Error).message}));
    } finally {setBusy('');}
  };
  return <details><summary>素材平台 API Key 与申请入口</summary>
    <p>素材 Key 用于官方搜索，与 Qwen 模型 Key 分开配置。软件按文稿需要搜索并缓存结果；验证只查询素材，不调用模型。</p>
    {providers.map(provider=>{
      const key=provider.id+'_api_key';
      return <div key={provider.id}>
        <label>{provider.name} API Key（可选）<input type="password" autoComplete="off" value={values[key]||''} placeholder={values[key+'_set']?'已保存，留空保留原密钥':'填写 API Key'} disabled={!!busy} onChange={event=>{change(key,event.target.value);setMessages(current=>({...current,[provider.id]:''}))}}/></label>
        <p>{provider.hint}</p>
        <div className="toolbar-actions">
          <a className="secondary small" href={provider.url} target="_blank" rel="noreferrer"><ExternalLink size={14}/>官方申请说明</a>
          <button className="secondary small" disabled={!!busy||(!values[key]?.trim()&&!values[key+'_set'])} onClick={()=>verify(provider.id)}>{busy===provider.id&&<Loader2 className="spin" size={14}/>}验证 Key</button>
        </div>
        {messages[provider.id]&&<p role="status">{messages[provider.id]}</p>}
      </div>;
    })}
    <p>Commons、NASA 无需素材 Key。Coverr 需先向官方说明应用用途并申请 API 接入。</p>
    <a href="https://api.coverr.co/docs/auth/" target="_blank" rel="noreferrer">Coverr 官方申请说明</a>
  </details>;
}
