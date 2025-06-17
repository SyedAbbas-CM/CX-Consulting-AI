/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  output: 'export',
  trailingSlash: true,
  // Removed rewrites as proxying is now handled by src/pages/api/[...proxy].ts
  // async rewrites() {
  //   return [
  //     {
  //       source: '/api/:path*', // Match any path starting with /api/
  //       destination: 'http://localhost:8000/api/:path*', // Proxy to backend
  //     },
  //   ]
  // },
}

export default nextConfig
