const WebSocket = require('ws');

// Create a WebSocket server on port 8765
const wss = new WebSocket.Server({ port: 8765 });

console.log('WebSocket server running on ws://localhost:8765');

// Key code mapping
const KEYCODES = {
  'LEFT': 37,
  'RIGHT': 39,
  'FIRE': 32, // SPACE
  'ENTER': 13
};

// Track latest game state from browser
let latestGameState = null;

// When a client connects
wss.on('connection', (ws) => {
  console.log('Client connected');

  // Handle messages from clients
  ws.on('message', (message) => {
    const msgStr = message.toString().trim();

    // Check if message is JSON (game state from browser)
    if (msgStr.startsWith('{')) {
      try {
        const data = JSON.parse(msgStr);
        if (data.type === 'game_state') {
          latestGameState = data;
          // Forward game state to all non-browser clients (Python agent)
          wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify(data));
            }
          });
          return;
        }
      } catch (e) {}
    }

    const command = msgStr.toUpperCase();
    console.log(`Received command: ${command}`);

    // Check if the command is valid
    if (KEYCODES[command]) {
      const keyCode = KEYCODES[command];

      // Send keydown event
      const keydownEvent = JSON.stringify({
        type: 'keydown',
        keyCode: keyCode
      });

      // Broadcast keydown to all connected clients
      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(keydownEvent);
        }
      });

      // After a small delay, send the keyup event
      setTimeout(() => {
        const keyupEvent = JSON.stringify({
          type: 'keyup',
          keyCode: keyCode
        });

        // Broadcast keyup to all connected clients
        wss.clients.forEach((client) => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(keyupEvent);
          }
        });
      }, 100); // 100ms delay
    } else {
      console.log(`Unknown command: ${command}`);
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
}); 