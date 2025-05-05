import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const COLORS = {
  background: '#181F2A',
  userBubble: '#4F8EF7',
  assistantBubble: '#232B3E',
  userText: '#fff',
  assistantText: '#E0E6F6',
  border: '#2C3654',
  inputBg: '#232B3E',
  inputText: '#fff',
  buttonBg: '#4F8EF7',
  buttonText: '#fff',
  shadow: '0 2px 12px rgba(0,0,0,0.15)'
};

function formatLLMContent(content) {
  return <ReactMarkdown>{content}</ReactMarkdown>;
}

function ChatbotUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [devPrompt, setDevPrompt] = useState(null);
  const [showPrompt, setShowPrompt] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    inputRef.current?.focus();
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const res = await axios.post('/chat', {
        message: input,
        history: [...messages, userMsg],
        size: 5
      });
      const reply = res.data.reply;
      setMessages(prev => [...prev, { role: 'assistant', content: reply }]);
      if (res.data.prompt) {
        setDevPrompt(res.data.prompt);
        setShowPrompt(true);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: Could not get response.' }]);
    }
    setLoading(false);
  };

  const handleInputKeyDown = e => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(120deg, #1a2236 0%, #4F8EF7 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
      padding: 0,
      margin: 0,
      boxSizing: 'border-box',
      overflow: 'auto',
    }}>
      <div style={{
        background: 'rgba(24,31,42,0.92)',
        borderRadius: 32,
        boxShadow: '0 12px 48px 0 rgba(79,142,247,0.18), 0 1.5px 8px 0 rgba(44,54,84,0.10)',
        width: '100%',
        maxWidth: 620,
        minHeight: 640,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        border: `2.5px solid #4F8EF7`,
        backdropFilter: 'blur(16px)',
        margin: '32px 0',
      }}>
        <div style={{
          background: 'linear-gradient(90deg, #4F8EF7 0%, #232B3E 100%)',
          color: '#fff',
          padding: '32px 0 28px 0',
          textAlign: 'center',
          fontWeight: 900,
          fontSize: 32,
          letterSpacing: 2,
          borderBottom: `1.5px solid #2C3654`,
          boxShadow: '0 2px 12px rgba(79,142,247,0.10)',
          textShadow: '0 2px 8px #232B3E44',
        }}>
          <span style={{verticalAlign:'middle',display:'inline-block',marginRight:8}}>
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFD700" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{verticalAlign:'middle'}}>
              <rect x="3" y="8" width="18" height="10" rx="4" />
              <circle cx="7" cy="12" r="1" />
              <circle cx="17" cy="12" r="1" />
              <path d="M8 16v1a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-1" />
              <path d="M12 8V5m0-2v2" />
            </svg>
          </span>
          <span style={{color:'#FFD700'}}>Log Analytics Chatbot</span>
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '38px 36px 38px 36px',
          background: 'transparent',
          display: 'flex',
          flexDirection: 'column',
          gap: 28,
          height: 0,
          minHeight: 0,
          maxHeight: 480,
          scrollbarWidth: 'thin',
        }}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                alignItems: 'flex-end',
                width: '100%',
              }}
            >
              <div
                style={{
                  background: msg.role === 'user' ? 'linear-gradient(90deg, #4F8EF7 60%, #7FBCFF 100%)' : 'linear-gradient(90deg, #232B3E 60%, #4F8EF7 100%)',
                  color: msg.role === 'user' ? '#fff' : '#FFD700',
                  borderRadius: 22,
                  borderBottomRightRadius: msg.role === 'user' ? 8 : 22,
                  borderBottomLeftRadius: msg.role === 'user' ? 22 : 8,
                  padding: '18px 26px',
                  maxWidth: '80%',
                  fontSize: 18,
                  boxShadow: msg.role === 'user' ? '0 4px 16px rgba(79,142,247,0.18)' : '0 4px 16px rgba(44,54,84,0.18)',
                  border: msg.role === 'user' ? '2px solid #7FBCFF' : '2px solid #FFD700',
                  transition: 'background 0.2s',
                  wordBreak: 'break-word',
                  lineHeight: 1.7,
                  backdropFilter: 'blur(2px)',
                }}
              >
                <b style={{ fontWeight: 800, fontSize: 15, opacity: 0.85, letterSpacing: 0.5 }}>
                  {msg.role === 'user' ? 'You' : 'Assistant'}:
                </b>
                <div style={{ marginTop: 8, whiteSpace: 'pre-line' }}>
                  {msg.role === 'assistant' ? formatLLMContent(msg.content) : msg.content}
                </div>
                {msg.content === 'Note that you asked this question twice, but the answer remains the same!' && (
                  <div style={{ marginTop: 10, color: '#FFD700', fontWeight: 700, fontSize: 15 }}>
                    <span role="img" aria-label="info">ℹ️</span> Note that you asked this question twice, but the answer remains the same!
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', width: '100%' }}>
              <div style={{
                background: 'linear-gradient(90deg, #232B3E 60%, #4F8EF7 100%)',
                color: '#FFD700',
                borderRadius: 22,
                borderBottomLeftRadius: 8,
                padding: '18px 26px',
                fontSize: 18,
                maxWidth: '80%',
                fontStyle: 'italic',
                opacity: 0.7,
                border: '2px solid #FFD700',
                boxShadow: '0 4px 16px rgba(44,54,84,0.18)',
                backdropFilter: 'blur(2px)',
              }}>
                Assistant is typing...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          padding: '28px 28px 28px 28px',
          background: 'rgba(24,31,42,0.98)',
          borderTop: `1.5px solid #2C3654`,
          gap: 18,
        }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            style={{
              flex: 1,
              background: 'rgba(35,43,62,0.98)',
              color: '#fff',
              border: 'none',
              borderRadius: 18,
              padding: '18px 22px',
              fontSize: 19,
              outline: 'none',
              marginRight: 0,
              boxShadow: '0 2px 8px rgba(44,54,84,0.12)',
              caretColor: '#FFD700',
              fontFamily: 'Fira Mono, monospace',
              letterSpacing: 0.7,
              transition: 'box-shadow 0.2s',
            }}
            placeholder="Ask about your logs..."
            disabled={loading}
            autoFocus
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              background: 'linear-gradient(90deg, #FFD700 0%, #4F8EF7 100%)',
              color: '#232B3E',
              border: 'none',
              borderRadius: 18,
              padding: '18px 36px',
              fontWeight: 900,
              fontSize: 19,
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 16px rgba(79,142,247,0.18)',
              transition: 'background 0.2s, box-shadow 0.2s, opacity 0.2s',
              opacity: loading || !input.trim() ? 0.7 : 1,
              fontFamily: 'Fira Mono, monospace',
              letterSpacing: 0.7,
              outline: 'none',
            }}
            onMouseOver={e => e.currentTarget.style.boxShadow = '0 8px 24px rgba(255,215,0,0.18)'}
            onMouseOut={e => e.currentTarget.style.boxShadow = '0 4px 16px rgba(79,142,247,0.18)'}
          >
            <span style={{verticalAlign:'middle',display:'inline-block',marginRight:8}}>
              <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#232B3E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 2 11 13" />
                <path d="M22 2 15 22 11 13 2 9l20-7z" />
              </svg>
            </span>
            Send
          </button>
        </div>
      </div>
      <div style={{ color: '#FFD700', marginTop: 32, fontSize: 16, opacity: 0.92, fontWeight: 700, letterSpacing: 0.7, textShadow: '0 1px 4px #232B3E44' }}>
        Powered by <b>SimpleLLM</b> &middot; <span role="img" aria-label="log">📝</span>
      </div>
      {showPrompt && devPrompt && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(24,31,42,0.85)',
          zIndex: 9999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            background: '#232B3E',
            color: '#FFD700',
            borderRadius: 18,
            padding: 32,
            maxWidth: 800,
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 8px 32px rgba(44,54,84,0.28)',
            fontFamily: 'Fira Mono, monospace',
            fontSize: 15,
            position: 'relative',
          }}>
            <button onClick={() => setShowPrompt(false)} style={{
              position: 'absolute',
              top: 12,
              right: 18,
              background: '#FFD700',
              color: '#232B3E',
              border: 'none',
              borderRadius: 8,
              padding: '6px 16px',
              fontWeight: 900,
              fontSize: 16,
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(44,54,84,0.18)',
            }}>Close</button>
            <div style={{ marginBottom: 18, fontWeight: 800, fontSize: 18 }}>LLM Prompt (Dev View)</div>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#FFD700', background: 'none', fontFamily: 'Fira Mono, monospace', fontSize: 14 }}>{devPrompt}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatbotUI;
