# Prerequisites

Before setting up Claude Context, ensure you have the following requirements met.

## Required Services

### Embedding Provider (Choose One)

#### Option 1: OpenAI (Recommended)
- **API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Billing**: Active billing account required
- **Models**: `text-embedding-3-small` or `text-embedding-3-large`
- **Rate Limits**: Check current limits on your OpenAI account

#### Option 2: VoyageAI
- **API Key**: Get from [VoyageAI Console](https://dash.voyageai.com/)
- **Models**: `voyage-code-3` (optimized for code)
- **Billing**: Pay-per-use pricing

#### Option 3: Gemini
- **API Key**: Get from [Google AI Studio](https://aistudio.google.com/)
- **Models**: `gemini-embedding-001`, `gemini-embedding-2`
- **Quota**: Check current quotas and limits

#### Option 4: Ollama (Local)
- **Installation**: Download from [ollama.com](https://ollama.com/)
- **Models**: Pull embedding models like `nomic-embed-text`
- **Hardware**: Sufficient RAM for model loading (varies by model)

### Vector Database

#### Zilliz Cloud (Recommended)
![](../../assets/signup_and_get_apikey.png)
- **Account**: [Sign up](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=2507-codecontext-readme) on Zilliz Cloud to get an API key.
- **Convenience**: Fully managed Milvus vector database service without the need to install and manage it.

#### Local Milvus (Advanced)
- **Docker**: Install Milvus by following [this guide](https://milvus.io/docs/install_standalone-docker-compose.md)
- **Resources**: More complex configuration required

#### Option 3: PostgreSQL + PGVector
- **PostgreSQL**: Version 12+ with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- **Setup**: Install pgvector extension by following [this guide](https://github.com/pgvector/pgvector#installation)
- **Configuration**: Set `VECTOR_DB_PROVIDER=pgvector` and `PG_CONNECTION_STRING=postgresql://user:pass@localhost:5432/dbname`
- **Note**: MCP server only; VS Code and Chrome extensions continue using Milvus

## Development Tools (Optional)

### For VSCode Extension
- **VSCode**: Version 1.74.0 or higher
- **Extensions**: Claude Context extension from marketplace


### For Development Contributions
- **Git**: For version control
- **pnpm**: Package manager (preferred over npm)
- **TypeScript**: Understanding of TypeScript development
