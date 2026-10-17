import type { Metadata } from 'next'
import { ThemeProvider } from 'next-themes'

import './globals.css'
import Providers from '@/app/providers'

export const metadata: Metadata = {
  title: 'Koharu',
  description:
    'Local-first comic and manga translation studio with OCR, inpainting, and typesetting.',
  openGraph: {
    title: 'Koharu',
    description:
      'Local-first comic and manga translation studio with OCR, inpainting, and typesetting.',
  },
}

function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang='en-US' suppressHydrationWarning style={{ backgroundColor: 'transparent' }}>
      <body className='antialiased' style={{ backgroundColor: 'transparent' }}>
        <ThemeProvider attribute='class' defaultTheme='system' enableSystem>
          <Providers>{children}</Providers>
        </ThemeProvider>
      </body>
    </html>
  )
}

export default RootLayout
