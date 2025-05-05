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
      background: 'linear-gradient(135deg, #232B3E 0%, #4F8EF7 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Segoe UI, Arial, sans-serif',
      padding: 0,
      margin: 0
    }}>
      <div style={{
        background: 'rgba(24,31,42,0.98)',
        borderRadius: 22,
        boxShadow: '0 8px 32px 0 rgba(79,142,247,0.25)',
        width: '100%',
        maxWidth: 520,
        minHeight: 540,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        border: `2.5px solid #4F8EF7`,
        backdropFilter: 'blur(8px)'
      }}>
        <div style={{
          background: 'linear-gradient(90deg, #4F8EF7 0%, #232B3E 100%)',
          color: '#fff',
          padding: '22px 0',
          textAlign: 'center',
          fontWeight: 800,
          fontSize: 26,
          letterSpacing: 1.5,
          borderBottom: `1px solid #2C3654`,
          boxShadow: '0 2px 8px rgba(79,142,247,0.08)'
        }}>
          <span role="img" aria-label="bot">🤖</span> <span style={{color:'#FFD700'}}>Log Analytics Chatbot</span>
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: 24,
          background: 'transparent',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
          height: 0,
          minHeight: 0,
          maxHeight: 400
        }}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                alignItems: 'flex-end'
              }}
            >
              <div
                style={{
                  background: msg.role === 'user' ? 'linear-gradient(90deg, #4F8EF7 60%, #7FBCFF 100%)' : 'linear-gradient(90deg, #232B3E 60%, #4F8EF7 100%)',
                  color: msg.role === 'user' ? '#fff' : '#FFD700',
                  borderRadius: 18,
                  borderBottomRightRadius: msg.role === 'user' ? 6 : 18,
                  borderBottomLeftRadius: msg.role === 'user' ? 18 : 6,
                  padding: '12px 18px',
                  maxWidth: '75%',
                  fontSize: 16,
                  boxShadow: msg.role === 'user' ? '0 2px 8px rgba(79,142,247,0.18)' : '0 2px 8px rgba(44,54,84,0.18)',
                  border: msg.role === 'user' ? '1.5px solid #7FBCFF' : '1.5px solid #FFD700',
                  transition: 'background 0.2s'
                }}
              >
                <b style={{ fontWeight: 700, fontSize: 13, opacity: 0.8 }}>
                  {msg.role === 'user' ? 'You' : 'Assistant'}:
                </b>
                <div style={{ marginTop: 4, whiteSpace: 'pre-line' }}>
                  {msg.role === 'assistant' ? formatLLMContent(msg.content) : msg.content}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
              <div style={{
                background: 'linear-gradient(90deg, #232B3E 60%, #4F8EF7 100%)',
                color: '#FFD700',
                borderRadius: 18,
                borderBottomLeftRadius: 6,
                padding: '12px 18px',
                fontSize: 16,
                maxWidth: '75%',
                fontStyle: 'italic',
                opacity: 0.7,
                border: '1.5px solid #FFD700'
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
          padding: 18,
          background: 'rgba(24,31,42,0.98)',
          borderTop: `1px solid #2C3654`
        }}>
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            style={{
              flex: 1,
              background: '#232B3E',
              color: '#fff',
              border: 'none',
              borderRadius: 14,
              padding: '14px 18px',
              fontSize: 17,
              outline: 'none',
              marginRight: 14,
              boxShadow: '0 1px 4px rgba(44,54,84,0.12)',
              caretColor: '#FFD700',
              fontFamily: 'Fira Mono, monospace',
              letterSpacing: 0.5
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
              borderRadius: 14,
              padding: '14px 26px',
              fontWeight: 700,
              fontSize: 17,
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              boxShadow: '0 2px 12px rgba(79,142,247,0.18)',
              transition: 'background 0.2s',
              opacity: loading || !input.trim() ? 0.7 : 1,
              fontFamily: 'Fira Mono, monospace',
              letterSpacing: 0.5
            }}
          >
            <span role="img" aria-label="send">📤</span> Send
          </button>
        </div>
      </div>
      <div style={{ color: '#FFD700', marginTop: 18, fontSize: 14, opacity: 0.85, fontWeight: 600, letterSpacing: 0.5 }}>
        Powered by <b>SimpleLLM</b> &middot; <span role="img" aria-label="log">📝</span>
      </div>
    </div>
  );
}

export default ChatbotUI;
