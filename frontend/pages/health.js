// Health check endpoint for frontend
export default function Health() {
  return null;
}

export async function getServerSideProps() {
  return {
    props: {},
  };
}

// API route for health check
export const config = {
  api: {
    bodyParser: false,
  },
};

// Health check handler
export async function handler(req, res) {
  if (req.method === 'GET') {
    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'financial-audit-frontend',
      version: process.env.npm_package_version || '1.0.0',
    });
  } else {
    res.setHeader('Allow', ['GET']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}