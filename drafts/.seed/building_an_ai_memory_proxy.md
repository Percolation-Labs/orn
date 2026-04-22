# Building an AI Memory Proxy

In Percolate we use many different patterns for wiring together LLMs, data and clients for different contexts. One way I like to think about Percolate is as a **memory proxy**. While much of this will sound similar to what you can do with any agent framework, I thought I would share some of the snippets I used to do this.

The _memory proxy_ pattern aims to be a simple conduit relaying messages essentially "as is" between clients and language models, while injecting a persistence capability for storing interactions, modeling user's (preferences, conversation history etc.), providing access to documents and data etc. There are many ways to implement those specific things. But I wanted to briefly mention today is a gateway that respects various interfaces while doing this.

We want to given the illusion we are just talking to the language model API in whatever dialect we choose. But the proxy intercepts the messages and injects data, functions, and tracking capabilities, while not changing the interface. This is what I call the _Memory Proxy pattern._

In Percolate there are two primary completions endpoints, one of which I discussed in a previous article showing how we can use OpenWebUI (or any client that speaks the OpenAI protocol) to connect to Percolate. Percolate allows for us to store the configuration of any model or functions as data and interact with agents via an OpenAPI like interface.

The agent completion endpoint `/agents/{agent_name}/completions` provides a full agentic loop but still provides the same interface (by default the OpenAPI one but also the Anthropic or Google one). We use a simple adapter which streams _content_ as-is to the client but buffers function calls, invokes functions, iterates on the loop exchanging tool data with the language model and continues to flush the content stream.

> While simple in principle, this is useful for building clients such as mobile applications where we can connect to a well defined interface and stream data without worrying about complex multi-hop iterative loops or data lookups on the server.

When doing function calling, streaming is not useful so we want a sort of hybrid streaming. When we are updating the user we should stream content, otherwise we should buffer a function, invoke it, have the LLM interpret it, iterate and then stream to the user.

In Percolate we have a ModelRunner that runs any Pydantic-based agent (and there is an equivalent model runner in the database). To implement the stream with tool call "buffering" we use the utility above and then below we provide an Iterator that "looks like" a streaming HTTP response.

This is an excerpt from the ModelRunner in Percolate and we can use this pattern to expose a completions endpoint in the API that exposes any Agent (which can use tools etc.) while streaming to the client in a consistent way.

In my case this pattern was very convenient. I have a mobile app that I was connecting to different language models and I could simply swap out the regular completions endpoint with this agentic one and on the surface everything looks the same. I spent half a day playing with the interface in a way that bubbles up nicely to the FastAPI endpoint so I thought I would share the snippets.

Check out the Percolate repo below (and give us some stars if you find it interesting).
