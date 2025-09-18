'use client'

import {
  ChartBarIcon,
  DocumentTextIcon,
  ClipboardDocumentListIcon,
  CogIcon
} from '@heroicons/react/24/outline'

interface SidebarProps {
  activePage: string
  onPageChange: (page: string) => void
}

const navigation = [
  { name: 'Dashboard', id: 'dashboard', icon: ChartBarIcon },
  { name: 'Audits', id: 'audits', icon: ClipboardDocumentListIcon },
  { name: 'Reports', id: 'reports', icon: DocumentTextIcon },
  { name: 'Settings', id: 'settings', icon: CogIcon },
]

export default function Sidebar({ activePage, onPageChange }: SidebarProps) {
  return (
    <div className="flex flex-col w-64 bg-gray-800">
      <div className="flex items-center h-16 px-4 bg-gray-900">
        <h1 className="text-white text-lg font-semibold">
          Financial Audit System
        </h1>
      </div>

      <nav className="mt-5 flex-1 px-2 space-y-1">
        {navigation.map((item) => {
          const isActive = activePage === item.id
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`
                group flex items-center w-full px-2 py-2 text-sm font-medium rounded-md
                ${isActive
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }
              `}
            >
              <item.icon
                className={`
                  mr-3 h-6 w-6
                  ${isActive ? 'text-gray-300' : 'text-gray-400 group-hover:text-gray-300'}
                `}
                aria-hidden="true"
              />
              {item.name}
            </button>
          )
        })}
      </nav>
    </div>
  )
}