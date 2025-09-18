'use client'

import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon, DocumentIcon, XMarkIcon, CheckCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import { documentApi, Document } from '../lib/api'
import { useMutation, useQueryClient } from 'react-query'

interface DocumentUploadProps {
  auditId: string
  onUploadComplete?: (document: Document) => void
  onUploadError?: (error: string) => void
  maxFiles?: number
  maxFileSize?: number
  acceptedFileTypes?: string[]
  className?: string
}

interface UploadingFile {
  id: string
  file: File
  progress: number
  status: 'uploading' | 'completed' | 'error'
  error?: string
}

const defaultAcceptedTypes = [
  'application/pdf',
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/csv',
  'application/json',
  'text/plain',
  'text/markdown',
  'application/msword'
]

export default function DocumentUpload({
  auditId,
  onUploadComplete,
  onUploadError,
  maxFiles = 10,
  maxFileSize = 50 * 1024 * 1024, // 50MB
  acceptedFileTypes = defaultAcceptedTypes,
  className = ''
}: DocumentUploadProps) {
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const uploadMutation = useMutation(
    async ({ file, auditId }: { file: File; auditId: string }) => {
      return documentApi.uploadDocument(auditId, file, (progress) => {
        setUploadingFiles(prev =>
          prev.map(f =>
            f.file === file ? { ...f, progress } : f
          )
        )
      })
    },
    {
      onSuccess: (document, { file }) => {
        setUploadingFiles(prev =>
          prev.map(f =>
            f.file === file
              ? { ...f, status: 'completed', progress: 100 }
              : f
          )
        )
        queryClient.invalidateQueries(['documents', auditId])
        onUploadComplete?.(document)

        // Remove completed file after 3 seconds
        setTimeout(() => {
          setUploadingFiles(prev => prev.filter(f => f.file !== file))
        }, 3000)
      },
      onError: (error: any, { file }) => {
        const errorMessage = error.message || 'Upload failed'
        setUploadingFiles(prev =>
          prev.map(f =>
            f.file === file
              ? { ...f, status: 'error', error: errorMessage }
              : f
          )
        )
        onUploadError?.(errorMessage)
      }
    }
  )

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      const errorMessage = errors.map((e: any) => e.message).join(', ')
      onUploadError?.(`${file.name}: ${errorMessage}`)\n    })

    // Process accepted files
    acceptedFiles.forEach(file => {
      const uploadingFile: UploadingFile = {
        id: Math.random().toString(36).substr(2, 9),
        file,
        progress: 0,
        status: 'uploading'
      }

      setUploadingFiles(prev => [...prev, uploadingFile])
      uploadMutation.mutate({ file, auditId })
    })
  }, [auditId, uploadMutation, onUploadError])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    maxFiles,
    maxSize: maxFileSize,
    accept: acceptedFileTypes.reduce((acc, type) => {
      acc[type] = []
      return acc
    }, {} as Record<string, string[]>),
    multiple: true
  })

  const removeUploadingFile = (id: string) => {
    setUploadingFiles(prev => prev.filter(f => f.id !== id))
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (file: File) => {
    const extension = file.name.split('.').pop()?.toLowerCase()
    switch (extension) {
      case 'pdf':
        return '📄'
      case 'xlsx':
      case 'xls':
        return '📊'
      case 'docx':
      case 'doc':
        return '📝'
      case 'csv':
        return '📈'
      case 'json':
        return '📋'
      case 'txt':
        return '📄'
      case 'md':
      case 'markdown':
        return '📝'
      default:
        return '📁'
    }
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive && !isDragReject ? 'border-blue-400 bg-blue-50' : ''}
          ${isDragReject ? 'border-red-400 bg-red-50' : ''}
          ${!isDragActive ? 'border-gray-300 hover:border-gray-400' : ''}
        `}
      >
        <input {...getInputProps()} ref={fileInputRef} />

        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />

        <div className="mt-4">
          {isDragActive ? (
            isDragReject ? (
              <p className="text-red-600 font-medium">
                Some files are not supported
              </p>
            ) : (
              <p className="text-blue-600 font-medium">
                Drop files here to upload
              </p>
            )
          ) : (
            <>
              <p className="text-lg font-medium text-gray-900">
                Drag and drop files here, or click to browse
              </p>
              <p className="text-sm text-gray-600 mt-2">
                Supports PDF, Word, Excel, CSV, JSON, Markdown, and text files up to {formatFileSize(maxFileSize)}
              </p>
            </>
          )}
        </div>

        <div className="mt-4 flex justify-center">
          <button
            type="button"
            className="btn-primary"
            onClick={() => fileInputRef.current?.click()}
          >
            Select Files
          </button>
        </div>
      </div>

      {/* Upload progress */}
      {uploadingFiles.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900">Uploading Files</h4>
          {uploadingFiles.map((uploadingFile) => (
            <div
              key={uploadingFile.id}
              className="bg-white border rounded-lg p-4 shadow-sm"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 flex-1">
                  <div className="text-2xl">
                    {getFileIcon(uploadingFile.file)}
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {uploadingFile.file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(uploadingFile.file.size)}
                    </p>
                  </div>

                  <div className="flex items-center space-x-2">
                    {uploadingFile.status === 'uploading' && (
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${uploadingFile.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600">
                          {uploadingFile.progress}%
                        </span>
                      </div>
                    )}

                    {uploadingFile.status === 'completed' && (
                      <CheckCircleIcon className="h-5 w-5 text-green-500" />
                    )}

                    {uploadingFile.status === 'error' && (
                      <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                    )}

                    <button
                      onClick={() => removeUploadingFile(uploadingFile.id)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>

              {uploadingFile.status === 'error' && uploadingFile.error && (
                <div className="mt-2 text-sm text-red-600">
                  {uploadingFile.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* File type info */}
      <div className="text-xs text-gray-500">
        <p>Supported file types:</p>
        <ul className="mt-1 space-y-1">
          <li>• PDF documents (.pdf)</li>
          <li>• Word documents (.docx, .doc)</li>
          <li>• Excel spreadsheets (.xlsx, .xls)</li>
          <li>• CSV files (.csv)</li>
          <li>• JSON data files (.json)</li>
          <li>• Text files (.txt)</li>
          <li>• Markdown files (.md, .markdown)</li>
        </ul>
      </div>
    </div>
  )
}