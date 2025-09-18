'use client'

import { Inter } from 'next/font/google'
import { QueryClient, QueryClientProvider } from 'react-query'
import { useState } from 'react'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [queryClient] = useState(() => new QueryClient())

  return (
    <html lang="en">
      <body className={inter.className}>
        <QueryClientProvider client={queryClient}>
          <div className="min-h-screen bg-gray-50">
            {children}
          </div>
        </QueryClientProvider>
      </body>
    </html>
  )
}