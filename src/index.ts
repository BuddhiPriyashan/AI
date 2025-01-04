
import * as z from 'zod';

// Import the Genkit core libraries and plugins.
import { generate } from '@genkit-ai/ai';
import { configureGenkit } from '@genkit-ai/core';
import { defineFlow, startFlowsServer } from '@genkit-ai/flow';
import { vertexAI } from '@genkit-ai/vertexai';
import { defineDotprompt, dotprompt } from '@genkit-ai/dotprompt';

// Import models from the Vertex AI plugin. The Vertex AI API provides access to
// several generative models. Here, we import Gemini 1.5 Flash.
import { gemini15Flash } from '@genkit-ai/vertexai';

configureGenkit({
  plugins: [
    // Load the Vertex AI plugin. You can optionally specify your project ID
    // by passing in a config object; if you don't, the Vertex AI plugin uses
    // the value from the GCLOUD_PROJECT environment variable.
    vertexAI({ location: 'us-east1' }),
  ],
  // Log debug output to tbe console.
  logLevel: 'debug',
  // Perform OpenTelemetry instrumentation and enable trace collection.
  enableTracingAndMetrics: true,
});

// Define a simple flow that prompts an LLM to generate menu suggestions.
export const menuSuggestionFlow = defineFlow(
  {
    name: 'menuSuggestionFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
		// Construct a request and send it to the model API.
    const llmResponse = await generate({
      prompt: `Suggest an item for the menu of a ${subject} themed restaurant`,
      model: gemini15Flash,
      config: {
        temperature: 1,
      },
    });

		// Handle the response from the model API. In this sample, we just convert
    // it to a string, but more complicated flows might coerce the response into
    // structured output or chain the response into another LLM call, etc.
    return llmResponse.text();
  }
);

// Define a new flow called jokeFlow
export const jokeFlow = defineFlow(
  {
      name: 'jokeFlow',
      inputSchema: z.string(),
      outputSchema: z.string(),
  },
  async (subject) => {
      const llmResponse = await generate({
          prompt: `Tell me a joke about ${subject}`,
          model: gemini15Flash,
          config: {
          temperature: 1,
          },
      });
  
      return llmResponse.text();
  }
);

//New flow with structured prompts
//(1)Adding input schema
const CustomerTimeAndHistorySchema = z.object({
  customerName: z.string(),
  currentTime: z.string(),
  previousOrder: z.string(),
});

//(2) Prompt object that flow will use
//The prompt contains the input schema, the model to use, and the text prompt.
const greetingWithHistoryPrompt = defineDotprompt(
  {
      name: 'greetingWithHistory',
      model: gemini15Flash,
      input: { schema: CustomerTimeAndHistorySchema },
      output: {
          format: 'text',
      },
},
  `
  {{role "user"}}
  Hi, my name is {{customerName}}. The time is {{currentTime}}. Who are you?
  
  {{role "model"}}
  I am Barb, a barista at this nice underwater-themed coffee shop called Krabby Kooffee.
  I know pretty much everything there is to know about coffee,
  and I can cheerfully recommend delicious coffee drinks to you based on whatever you like.
  
  {{role "user"}}
  Great. Last time I had {{previousOrder}}.
  I want you to greet me in one sentence, and recommend a drink.
  `
);

//(3) Define the flow that contian the input schema and the defined prompt
export const greetingWithHistoryFlow = defineFlow(
  {
      name: 'greetingWithHistory',
      inputSchema: CustomerTimeAndHistorySchema,
      outputSchema: z.string(),
  },
  async (input) =>
      (await greetingWithHistoryPrompt.generate({ input: input })).text()
);

// Start a flow server, which exposes your flows as HTTP endpoints. This call
// must come last, after all of your plug-in configuration and flow definitions.
// You can optionally specify a subset of flows to serve, and configure some
// HTTP server options, but by default, the flow server serves all defined flows.
startFlowsServer();
