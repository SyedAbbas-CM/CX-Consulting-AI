import { createProxyMiddleware } from 'http-proxy-middleware'
import type { NextApiRequest, NextApiResponse } from 'next'

// Configuration for the Next.js API route
export const config = {
  api: {
    bodyParser: false, // Let the proxy handle the body
    externalResolver: true // Important for http-proxy-middleware
  },
}

// Create the proxy middleware instance
const proxy = createProxyMiddleware({
  target: 'http://localhost:8000',   // Your FastAPI backend URL
  changeOrigin: true,                // Recommended for virtual hosted sites
  pathRewrite: { '^/api': '/api' },  // Keep the /api prefix when forwarding
  timeout:      300_000,             // 5 minutes idle socket timeout (ms)
  proxyTimeout: 300_000,             // 5 minutes for the backend to respond (ms)
  // Removed optional logging handlers to resolve type errors
  // onError: (err, req, res) => { ... },
  // onProxyReq: (proxyReq, req, res) => { ... },
  // onProxyRes: (proxyRes, req, res) => { ... }
})

// The actual API route handler
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // Use a Promise to handle the proxy middleware which might resolve asynchronously
  return new Promise<void>((resolve, reject) => {
    // Handle potential request errors (e.g., client closes connection)
    req.on('error', (err) => {
      console.error('Request Error:', err);
      reject(err);
    });
    res.on('error', (err) => {
        console.error('Response Error:', err);
        reject(err);
    });

    // Execute the proxy middleware
    proxy(req, res, (result) => {
      if (result instanceof Error) {
        console.error('Proxy middleware callback error:', result);
        return reject(result);
      }
      return resolve();
    });
  });
}
