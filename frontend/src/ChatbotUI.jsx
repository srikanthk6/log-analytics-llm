import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css';

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
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{content}</ReactMarkdown>
  );
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
      height: '100vh',
      width: '100vw',
      background: '#f7f7f8',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
      margin: 0,
      boxSizing: 'border-box',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        background: '#fff',
        color: '#222',
        padding: '16px 0 12px 0',
        textAlign: 'center',
        fontWeight: 800,
        fontSize: 24,
        letterSpacing: 1.5,
        borderBottom: '1px solid #ececec',
        boxShadow: '0 1px 8px 0 rgba(0,0,0,0.03)',
        zIndex: 2,
      }}>
        <span style={{verticalAlign:'middle',display:'inline-block',marginRight:8}}>
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4F8EF7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{verticalAlign:'middle'}}>
            <rect x="3" y="8" width="18" height="10" rx="4" />
            <circle cx="7" cy="12" r="1" />
            <circle cx="17" cy="12" r="1" />
            <path d="M8 16v1a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-1" />
            <path d="M12 8V5m0-2v2" />
          </svg>
        </span>
        <span style={{color:'#4F8EF7'}}>Log Analytics Chatbot</span>
      </div>
      {/* Chat area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        background: '#f7f7f8',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        padding: '0 0 110px 0',
        scrollbarWidth: 'thin',
        minHeight: 0,
      }}>
        <div style={{padding: '18px 0 0 0', width: '100%', maxWidth: 720, margin: '0 auto'}}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                alignItems: 'flex-end',
                width: '100%',
                marginBottom: 6,
              }}
            >
              <div
                style={{
                  background: msg.role === 'user' ? '#fff' : '#f0f2f6',
                  color: '#222',
                  borderRadius: 10,
                  borderBottomRightRadius: msg.role === 'user' ? 4 : 10,
                  borderBottomLeftRadius: msg.role === 'user' ? 10 : 4,
                  padding: '10px 16px',
                  maxWidth: '80%',
                  fontSize: 16,
                  boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                  border: '1px solid #ececec',
                  transition: 'background 0.2s',
                  wordBreak: 'break-word',
                  lineHeight: 1.6,
                  margin: '0 6px',
                  backdropFilter: 'none',
                }}
              >
                <b style={{ fontWeight: 600, fontSize: 13, opacity: 0.7, letterSpacing: 0.3 }}>
                  {msg.role === 'user' ? 'You' : 'Assistant'}:
                </b>
                <div style={{ marginTop: 4, whiteSpace: 'pre-line' }}>
                  {msg.role === 'assistant' ? formatLLMContent(msg.content) : msg.content}
                </div>
                {msg.content === 'Note that you asked this question twice, but the answer remains the same!' && (
                  <div style={{ marginTop: 6, color: '#4F8EF7', fontWeight: 600, fontSize: 13 }}>
                    <span role="img" aria-label="info">ℹ️</span> Note that you asked this question twice, but the answer remains the same!
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', width: '100%' }}>
              <div style={{
                background: '#f0f2f6',
                color: '#222',
                borderRadius: 10,
                borderBottomLeftRadius: 4,
                padding: '10px 16px',
                fontSize: 16,
                maxWidth: '80%',
                fontStyle: 'italic',
                opacity: 0.7,
                border: '1px solid #ececec',
                boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                backdropFilter: 'none',
              }}>
                Assistant is typing...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>
      {/* Input area fixed at bottom */}
      <div style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        width: '100vw',
        background: '#fff',
        borderTop: '1px solid #ececec',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '14px 8px 14px 8px',
        zIndex: 10,
        boxShadow: '0 -1px 8px rgba(0,0,0,0.03)',
      }}>
        <div style={{width:'100%',maxWidth:720,margin:'0 auto',display:'flex',alignItems:'center',gap:10}}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            style={{
              flex: 1,
              background: '#f7f7f8',
              color: '#222',
              border: '1px solid #ececec',
              borderRadius: 8,
              padding: '12px 14px',
              fontSize: 16,
              outline: 'none',
              marginRight: 0,
              boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
              caretColor: '#4F8EF7',
              fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
              letterSpacing: 0.3,
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
              background: '#4F8EF7',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              padding: '12px 22px',
              fontWeight: 700,
              fontSize: 16,
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              boxShadow: '0 1px 4px rgba(79,142,247,0.10)',
              transition: 'background 0.2s, box-shadow 0.2s, opacity 0.2s',
              opacity: loading || !input.trim() ? 0.7 : 1,
              fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
              letterSpacing: 0.3,
              outline: 'none',
            }}
            onMouseOver={e => e.currentTarget.style.background = '#2563eb'}
            onMouseOut={e => e.currentTarget.style.background = '#4F8EF7'}
          >
            <span style={{verticalAlign:'middle',display:'inline-block',marginRight:6}}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 2 11 13" />
                <path d="M22 2 15 22 11 13 2 9l20-7z" />
              </svg>
            </span>
            Send
          </button>
        </div>
      </div>
      {/* Footer */}
      <div style={{ color: '#4F8EF7', marginTop: 8, fontSize: 14, opacity: 0.92, fontWeight: 600, letterSpacing: 0.3, textShadow: '0 1px 2px #fff', textAlign: 'center' }}>
        Powered by <b>SimpleLLM</b> &middot; <span role="img" aria-label="log">📝</span>
      </div>
      {/* Dev Prompt Modal */}
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
            background: '#fff',
            color: '#222',
            borderRadius: 10,
            padding: 24,
            maxWidth: 800,
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 8px 32px rgba(44,54,84,0.10)',
            fontFamily: 'Fira Mono, monospace',
            fontSize: 15,
            position: 'relative',
          }}>
            <button onClick={() => setShowPrompt(false)} style={{
              position: 'absolute',
              top: 12,
              right: 18,
              background: '#4F8EF7',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '6px 16px',
              fontWeight: 700,
              fontSize: 15,
              cursor: 'pointer',
              boxShadow: '0 1px 4px rgba(44,54,84,0.10)',
            }}>Close</button>
            <div style={{ marginBottom: 12, fontWeight: 700, fontSize: 16 }}>LLM Prompt (Dev View)</div>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#222', background: 'none', fontFamily: 'Fira Mono, monospace', fontSize: 14 }}>{devPrompt}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatbotUI;
