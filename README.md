# OpenAIVectorSearchDemo

### About:
This repo provides a function app that generates a list of text and a list of embeddings per each document found in Cognitive Search.  This is helpful when leveraging Azure Gov as it does not yet include Semantic Search.

0. Create a .env file

```
AZURE_OPENAI_ENDPOINT = "https://blah.openai.azure.com/"
AZURE_OPENAI_KEY = "123456678"

TEXT_EMBEDDING_ENGINE = "text-embedding-ada-002"
COG_SEARCH_RESOURCE = "mm-cogsearch-resource"
COG_SEARCH_KEY = "123456789"
COG_SEARCH_INDEX = "test-index2"

# #function app configuration
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=earchstorage;AccountKey=123455L/g==;EndpointSuffix=core.windows.net"
STORAGE_ACCOUNT = "searchstorage"
STORAGE_CONTAINER = "somemoredocs"
STORAGE_KEY = "123455L/g=="
COG_SERVICE_KEY = "XXXXYYYYZZZ"
DEBUG = "1"
USGOV = True

functionAppUrlAndKey = "https://cog-search-functionapp.azurewebsites.net/api/Embeddings?code=1234566uzYSB4A=="
```

1. The Embeddings Function should be deployed and will be a custom skill you will integrate with your Cogitive Search Resource

2. Deploy the following resources:

- Cognitive Search
- Cognitive Services
- Function App
- Azure OpenAI

3. The Function App Can be deployed directly from VS Code.  If you have troubles with that, you can create an additional Azure Container Registry where you can push a docker image to and use it in your Function App configuration.

4. For Azure Gov deployments, update the property self.endpoint = "https://{}.search.windows.net/".format(self.service_name) in the class CogSearchHelper in the file cog_search.py