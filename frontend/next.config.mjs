/** @type {import('next').NextConfig} */
const nextConfig = {
  // ─── Compiler ───────────────────────────────────────────────────────────
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // ─── React ──────────────────────────────────────────────────────────────
  reactStrictMode: true,

  // ─── Images ─────────────────────────────────────────────────────────────
  images: {
    unoptimized: false,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },

  // ─── Output ─────────────────────────────────────────────────────────────
  output: 'standalone',

  // ─── Build settings ─────────────────────────────────────────────────────
  productionBrowserSourceMaps: false,
  swcMinify: true,

  // ─── Environment variables ──────────────────────────────────────────────
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
    NEXT_PUBLIC_APP_VERSION: '1.0.0',
  },

  // ─── Headers ────────────────────────────────────────────────────────────
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },

  // ─── Redirects ──────────────────────────────────────────────────────────
  async redirects() {
    return [
      {
        source: '/dashboard',
        destination: '/',
        permanent: false,
      },
    ];
  },

  // ─── Rewrites ───────────────────────────────────────────────────────────
  async rewrites() {
    return {
      beforeFiles: [
        // Proxy /api/* to backend
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;