import React, { useState, useRef, useEffect } from 'react';

// Minimal Chat SDK Demo-inspired UI
function ChatSDKDemoUI() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you with your logs today?' }
  ]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim()) return;
    setMessages(prev => [
      ...prev,
      { role: 'user', content: input },
      { role: 'assistant', content: 'This is a demo reply. (Integrate your backend here.)' }
    ]);
    setInput('');
  };

  const handleInputKeyDown = e => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div style={{
      height: '100vh',
      width: '100vw',
      background: 'linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%)',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
      margin: 0,
      boxSizing: 'border-box',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        background: 'rgba(255,255,255,0.95)',
        color: '#1a237e',
        padding: '18px 0 14px 0',
        textAlign: 'center',
        fontWeight: 800,
        fontSize: 26,
        letterSpacing: 1.5,
        borderBottom: '1px solid #e3e8f0',
        boxShadow: '0 1px 8px 0 rgba(0,0,0,0.03)',
        zIndex: 2,
      }}>
        <span style={{color:'#2563eb'}}>Chat SDK Demo UI</span>
      </div>
      {/* Chat area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        padding: '0 0 110px 0',
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
                  background: msg.role === 'user' ? '#2563eb' : '#fff',
                  color: msg.role === 'user' ? '#fff' : '#222',
                  borderRadius: 18,
                  padding: '12px 20px',
                  maxWidth: '75%',
                  fontSize: 17,
                  boxShadow: '0 1px 6px rgba(0,0,0,0.07)',
                  margin: '0 6px',
                  wordBreak: 'break-word',
                  lineHeight: 1.7,
                }}
              >
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
      </div>
      {/* Input area fixed at bottom */}
      <div style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        width: '100vw',
        background: 'rgba(255,255,255,0.97)',
        borderTop: '1px solid #e3e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
        boxShadow: '0 -1px 8px rgba(0,0,0,0.03)',
        padding: '0 0 0 0',
      }}>
        <div style={{width:'100%',maxWidth:720,margin:'0 auto',display:'flex',alignItems:'center',gap:10,padding:'18px 0'}}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            style={{
              flex: 1,
              background: '#f3f6fa',
              color: '#222',
              border: '1.5px solid #e3e8f0',
              borderRadius: 22,
              padding: '16px 54px 16px 18px',
              fontSize: 18,
              outline: 'none',
              boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
              caretColor: '#2563eb',
              fontFamily: 'Inter, Segoe UI, Arial, sans-serif',
              letterSpacing: 0.3,
              minHeight: 44,
              fontWeight: 400,
              transition: 'border 0.2s, box-shadow 0.2s',
            }}
            placeholder="Type a message..."
            autoFocus
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim()}
            style={{
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '50%',
              width: 44,
              height: 44,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(37,99,235,0.10)',
              cursor: !input.trim() ? 'not-allowed' : 'pointer',
              opacity: !input.trim() ? 0.7 : 1,
              transition: 'background 0.2s, box-shadow 0.2s, opacity 0.2s',
              outline: 'none',
              padding: 0,
            }}
            tabIndex={0}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2 11 13" />
              <path d="M22 2 15 22 11 13 2 9l20-7z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatSDKDemoUI;
