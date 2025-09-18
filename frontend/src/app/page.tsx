'use client'

import { useState } from 'react'
import AuditDashboard from '@/components/AuditDashboard'
import Sidebar from '@/components/Sidebar'

export default function HomePage() {
  const [activePage, setActivePage] = useState('dashboard')

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <AuditDashboard />
      case 'audits':
        return <div className="p-6">Audits List Page</div>
      case 'reports':
        return <div className="p-6">Reports Page</div>
      case 'settings':
        return <div className="p-6">Settings Page</div>
      default:
        return <AuditDashboard />
    }
  }

  return (
    <div className="flex h-screen">
      <Sidebar activePage={activePage} onPageChange={setActivePage} />
      <main className="flex-1 overflow-auto">
        {renderPage()}
      </main>
    </div>
  )
}