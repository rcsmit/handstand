
import React, { useState, useRef, useEffect } from 'react';
import { analyzeHandstand } from './services/gemini';
import { AppState } from './types';
import JointOverlay from './components/JointOverlay';

const App: React.FC = () => {
  const [hasApiKey, setHasApiKey] = useState<boolean>(true);
  const [checkingKey, setCheckingKey] = useState<boolean>(true);
  const [state, setState] = useState<AppState>({
    image: null,
    analyzing: false,
    result: null,
    error: null,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const checkKey = async () => {
      try {
        // Check both the aistudio helper and the injected process.env variable
        const studioCheck = await (window as any).aistudio?.hasSelectedApiKey();
        const envCheck = !!((window as any).process?.env?.API_KEY || (process as any)?.env?.API_KEY);
        
        // Ensure we handle the "must select key" flow if neither is found
        setHasApiKey(studioCheck || envCheck);
      } catch (e) {
        console.error("Key check failed", e);
        setHasApiKey(false);
      } finally {
        setCheckingKey(false);
      }
    };
    checkKey();
  }, []);

  const handleSelectKey = async () => {
    if ((window as any).aistudio?.openSelectKey) {
      await (window as any).aistudio.openSelectKey();
      // Assume success as per instructions to avoid race conditions
      setHasApiKey(true);
      setState(prev => ({ ...prev, error: null }));
    } else {
      setState(prev => ({ ...prev, error: "API Key Selection interface not found." }));
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
      setState(prev => ({ ...prev, error: "Image is too large. Please use a file under 10MB." }));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target?.result as string;
      setState({ image: base64, analyzing: false, result: null, error: null });
      processImage(base64);
    };
    reader.readAsDataURL(file);
  };

  const processImage = async (image: string) => {
    // Final safety check for API key before calling service
    const envCheck = !!((window as any).process?.env?.API_KEY || (process as any)?.env?.API_KEY);
    if (!envCheck && !hasApiKey) {
        setHasApiKey(false);
        return;
    }

    setState(prev => ({ ...prev, analyzing: true, error: null }));
    await new Promise(r => setTimeout(r, 400));

    try {
      const result = await analyzeHandstand(image);
      setState(prev => ({ ...prev, result, analyzing: false }));
    } catch (err: any) {
      console.error(err);
      if (err.message.includes("API Key issue") || err.message.includes("API key")) {
        setHasApiKey(false);
      }
      setState(prev => ({ 
        ...prev, 
        error: err.message || "Failed to analyze image. Please try again.", 
        analyzing: false 
      }));
    }
  };

  const reset = () => {
    setState({ image: null, analyzing: false, result: null, error: null });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  if (checkingKey) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-slate-500 font-medium animate-pulse">Initializing AI Studio...</p>
        </div>
      </div>
    );
  }

  if (!hasApiKey) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md w-full bg-white rounded-3xl shadow-2xl p-10 border border-slate-100 text-center animate-in fade-in zoom-in duration-300">
          <div className="bg-indigo-50 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8 border border-indigo-100">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
            </svg>
          </div>
          <h2 className="text-2xl font-black text-slate-900 mb-4 tracking-tight">API Key Required</h2>
          <p className="text-slate-600 mb-8 text-sm leading-relaxed font-medium">
            To analyze handstand form, you need to connect your Gemini API key from a paid project. 
            <br/><br/>
            Visit the <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" className="text-indigo-600 font-bold hover:underline" rel="noreferrer">Billing Docs</a> to set up your account.
          </p>
          <button 
            onClick={handleSelectKey}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-black py-4 rounded-2xl transition-all shadow-xl shadow-indigo-100 active:scale-[0.98] uppercase tracking-widest text-xs"
          >
            Connect Paid API Key
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 md:py-12">
      <header className="text-center mb-10">
        <h1 className="text-4xl md:text-6xl font-black bg-gradient-to-r from-indigo-700 to-emerald-600 bg-clip-text text-transparent mb-4 tracking-tight">
          HANDSTAND PRO AI
        </h1>
        <p className="text-slate-500 text-lg font-medium">Professional Stack & Torque Analysis</p>
      </header>

      {!state.image ? (
        <div 
          onClick={() => fileInputRef.current?.click()}
          className="border-4 border-dashed border-slate-200 bg-white rounded-[2.5rem] p-20 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50/20 transition-all group shadow-sm active:scale-[0.99]"
        >
          <div className="bg-slate-50 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8 group-hover:scale-110 transition-transform border border-slate-100 group-hover:bg-indigo-50 group-hover:border-indigo-100">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-slate-300 group-hover:text-indigo-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>
          <p className="text-2xl font-bold text-slate-800 mb-2 tracking-tight">Upload Your Handstand</p>
          <p className="text-slate-400 font-medium">Clear profile views provide the most accurate leverage data.</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden" 
            accept="image/*"
          />
        </div>
      ) : (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-1000">
          <div className="grid lg:grid-cols-12 gap-10 items-start">
            
            <div className="lg:col-span-7 space-y-6">
              <div className="relative rounded-[2rem] overflow-hidden bg-white shadow-2xl ring-1 ring-slate-200/50">
                <img src={state.image} alt="Handstand" className="w-full h-auto block" />
                {state.result && <JointOverlay analysis={state.result} />}
                {state.analyzing && (
                  <div className="absolute inset-0 bg-white/80 backdrop-blur-2xl flex flex-col items-center justify-center z-10">
                    <div className="w-20 h-20 border-[8px] border-indigo-600 border-t-transparent rounded-full animate-spin mb-8"></div>
                    <div className="text-center">
                        <p className="text-indigo-700 font-black tracking-[0.2em] text-xl animate-pulse uppercase mb-2">Calculating Torque</p>
                        <p className="text-slate-400 font-medium text-sm">Identifying joint centers and center of mass...</p>
                    </div>
                  </div>
                )}
              </div>

              {state.result && (
                <div className="bg-white p-10 rounded-[2rem] border border-slate-200 shadow-sm space-y-6">
                   <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3">
                    <div className="w-3 h-8 bg-emerald-500 rounded-full"></div>
                    Stack Alignment
                  </h3>
                  <p className="text-slate-600 leading-relaxed font-medium">
                    The vertical dashed <strong>Plumb Line</strong> represents your ideal center of gravity through the base. 
                    Horizontal offsets indicate structural leverage that increases muscular demand.
                  </p>
                  <div className="grid grid-cols-3 gap-8 pt-6 border-t border-slate-50">
                    <LeverageMetric label="Ankle Offset" value={state.result.torque.offsets.ankle} />
                    <LeverageMetric label="Hip Offset" value={state.result.torque.offsets.hip} />
                    <LeverageMetric label="Shoulder Offset" value={state.result.torque.offsets.shoulder} />
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-5 space-y-6">
              {state.error && (
                <div className="bg-rose-50 border border-rose-200 text-rose-600 p-6 rounded-3xl flex items-center gap-5 shadow-sm">
                  <div className="bg-rose-100 p-2 rounded-full">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <p className="font-bold text-sm leading-tight">{state.error}</p>
                </div>
              )}

              {state.result && (
                <div className="space-y-6">
                  <div className="bg-white p-10 rounded-[2rem] border border-slate-200 shadow-sm relative overflow-hidden">
                    <div className="relative z-10">
                      <div className="flex justify-between items-start mb-8">
                        <div>
                          <p className="text-slate-400 text-xs font-black uppercase tracking-[0.2em] mb-2">Stack Rating</p>
                          <h2 className="text-7xl font-black text-slate-900 leading-none tracking-tighter">{state.result.score}%</h2>
                        </div>
                        <span className={`px-5 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest ${
                          state.result.classification === 'Perfect' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                        }`}>
                          {state.result.classification}
                        </span>
                      </div>
                      <div className="w-full h-5 bg-slate-50 rounded-full overflow-hidden border border-slate-100 p-1">
                        <div 
                          className="h-full rounded-full bg-gradient-to-r from-indigo-600 via-blue-500 to-emerald-400 transition-all duration-1500 ease-out"
                          style={{ width: `${state.result.score}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <AngleStat label="Shoulder Opening" angle={state.result.angles.shoulder_opening} target={180} />
                    <AngleStat label="Hip Flexion" angle={state.result.angles.hip_alignment} target={180} />
                    <AngleStat label="Elbow Extension" angle={state.result.angles.elbow_extension} target={180} />
                    <AngleStat label="Knee Extension" angle={state.result.angles.knee_straightness} target={180} />
                  </div>

                  <div className="bg-slate-900 text-white p-10 rounded-[2.5rem] shadow-2xl space-y-8 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 blur-3xl rounded-full -mr-10 -mt-10"></div>
                    <h3 className="text-2xl font-black flex items-center gap-4 relative z-10">
                      <div className="p-2 bg-indigo-500 rounded-xl">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                      AI Coaching
                    </h3>
                    <ul className="space-y-6 relative z-10">
                      {state.result.feedback.map((item, i) => (
                        <li key={i} className="flex gap-5 text-slate-300 text-sm leading-relaxed items-start group">
                          <span className="w-6 h-6 rounded-lg bg-slate-800 flex items-center justify-center shrink-0 text-[10px] font-black text-indigo-400 border border-slate-700 group-hover:border-indigo-500 transition-colors">{i+1}</span>
                          <span className="group-hover:text-white transition-colors">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {!state.analyzing && (
                <button 
                  onClick={reset}
                  className="w-full bg-slate-900 hover:bg-black text-white font-black py-6 rounded-2xl transition-all shadow-xl active:scale-[0.97] tracking-widest text-xs uppercase"
                >
                  {state.image ? "Reset Analysis" : "Analyze Form"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const LeverageMetric: React.FC<{ label: string; value: number }> = ({ label, value }) => {
  const absVal = Math.abs(value);
  const color = absVal < 4 ? 'text-emerald-500' : absVal < 10 ? 'text-amber-500' : 'text-rose-500';
  return (
    <div className="text-center group">
      <p className="text-[10px] font-black uppercase text-slate-400 mb-2 tracking-widest group-hover:text-indigo-400 transition-colors">{label}</p>
      <div className={`text-3xl font-black ${color} flex items-center justify-center gap-0.5`}>
          {absVal.toFixed(1)}
          <span className="text-[10px] font-black opacity-40">PX</span>
      </div>
    </div>
  );
};

const AngleStat: React.FC<{ label: string; angle: number; target: number }> = ({ label, angle, target }) => {
  const diff = Math.abs(angle - target);
  const statusColor = diff < 10 ? 'text-emerald-600' : diff < 20 ? 'text-amber-600' : 'text-rose-600';
  const bgColor = diff < 10 ? 'bg-emerald-50' : diff < 20 ? 'bg-amber-50' : 'bg-rose-50';

  return (
    <div className={`p-6 rounded-3xl border border-transparent transition-all hover:scale-105 duration-300 ${bgColor}`}>
      <p className="text-slate-500 text-[10px] font-black uppercase mb-2 tracking-widest opacity-60">{label}</p>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-3xl font-black ${statusColor}`}>{Math.round(angle)}°</span>
        <span className="text-slate-400 text-[10px] font-black opacity-40">/ {target}°</span>
      </div>
    </div>
  );
};

export default App;
