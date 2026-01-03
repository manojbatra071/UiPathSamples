\# Privacy Policy for Orchestrator CorePilot



\*\*Last Updated: January 3, 2026\*\*



Orchestrator CorePilot ("the Extension") is committed to protecting your privacy. This policy explains how we handle data and why we require certain permissions.



\## 1. No Developer Data Collection

Orchestrator CorePilot does NOT collect, store, or transmit any of your personal data, Orchestrator logs, or credentials to the developer or any third-party "middle-man" servers. All operations happen directly between your browser and the services you authorize.



\## 2. Third-Party AI Services

The Extension allows you to connect to various AI providers (OpenAI, Gemini, Mistral, Hugging Face). 

\- When you use the AI chat or diagnostic features, the Extension sends only the necessary technical context (e.g., specific job logs or asset descriptions) directly to your chosen AI provider.

\- Any data sent is subject to the privacy policy of the respective AI provider you have configured.



\## 3. Data Storage

\- \*\*API Keys\*\*: Your API keys are stored exclusively in `chrome.storage.sync`, which is encrypted at rest by Google and synced across your authenticated browser instances.

\- \*\*Local Settings\*\*: Preferences like Dark Mode or refresh intervals are also stored locally.

\- We do not have access to this data.



\## 4. Permissions Disclosure

\- \*\*host\_permissions\*\*: Used to communicate directly with UiPath Orchestrator and your chosen AI APIs.

\- \*\*scripting\*\*: Used to inject the chat UI and format logs locally on your machine.

\- \*\*alarms\*\*: Used for the "Live Watchdog" feature to check job status periods.

\- \*\*notifications\*\*: Used to alert you about automation failures.



\## 5. Contact

For any questions regarding this policy, please open an issue on our official GitHub repository.

