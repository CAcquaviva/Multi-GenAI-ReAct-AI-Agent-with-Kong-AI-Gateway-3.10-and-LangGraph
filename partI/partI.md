# Build a Better ReAct AI Agent with Kong AI Gateway

This is part I of this series exploring how Kong AI Gateway can be used in an AI Agent development with LangGraph. The series comprises three parts:



* Basic ReAct AI Agent with Kong AI Gateway
* Single LLM ReAct AI Agent with Kong AI Gateway and LangGraph
* Multi-LLM ReAct AI Agent and Kong AI Gateway and LangGraph Server


## Introduction

To put it simply: ChatGPT is an AI agent.

As such, ChatGPT is a type of artificial intelligence system designed to perceive natural language input, process information, and respond accordingly. Anthropic's blog post, [“Building Effective Agents”](https://www.anthropic.com/engineering/building-effective-agents), has a nice and concise definition of AI Agents: “Agents are systems that leverage LLMs to autonomously coordinate their operations and tool interactions, exercising control over task execution.”

On the other hand, no single model has all the answers. A multi-GenAI Agent brings the best of multiple LLM, Audio, Video and Image models together. Instead of relying on one brain, this agent intelligently selects or coordinates between different language models like GPT-4, Claude, Mistral, and others, to deliver more accurate responses.

Ideally, we should isolate the AI Agent from the policies that drive the consumption of the models. Or, if you will, the AI Agent should rely on an external component, intelligent enough to understand what and how many models are available and, most importantly, decide which model should be called to address the incoming requests. For example, let's say you have multiple available models, each one of them trained for specific topics like Mathematics, Classical Music, etc. Another scenario would consider digesting a variety of inputs, including texts, images and audios. Moreover, models can be added or removed from your environment along the way, making AI agent development even more challenging.

This blog post explores how developers can leverage Kong AI Gateway to implement better and smarter AI agents using [LangGraph](https://www.langchain.com/langgraph). We are going to focus on LLM models but keep in mind, Kong AI Gateway will support other kinds of models along the way, including audio, images, etc. Moreover, it's important to note that LangGraph is an extensive AI Agent framework with lots of other features and capabilities. Refer to its [documentation portal](https://langchain-ai.github.io/langgraph/) to learn more about it.

Before we get started, it’s important to make an important comment: the agents we are going to create and run are introductory ones and are not meant to be applied for production environments. The main purpose is to demonstrate how to leverage the LangGraph and Kong AI Gateway capabilities we have available when creating an AI Agent.


### MCP (Model Context Protocol)

We have seen a lot of traction around the new protocol proposed last [November, 2024, by Anthropic](https://www.anthropic.com/news/model-context-protocol), called [MCP](https://modelcontextprotocol.io/) (Model Context Protocol). At the same time Kong launched last April, 2025 the new [Kong MCP Server](https://konghq.com/blog/product-releases/mcp-server). MCP contributes tremendously to AI Agent developments, however, since this is an introductory blog post, we are going to exercise the inclusion of MCP in this architecture, integrating with Kong and LangGraph, in the next one.


### Blog structure

The post is structured in following sections:

* Part I
    * Kong AI Gateway Introduction, Implementation Architecture and AI Agent fundamentals
    * Kong Version of a simple ReAct based AI Agent with no frameworks
* Part II
    * LangGraph fundamentals and basic AI Agent
    * Tools and Function Calling with Kong AI Gateway and Observability Layer
    * Single LLM AI Agent with Kong AI Gateway and LangGraph
* Part III
    * Multi LLM AI Agent
    * LangGraph Server


## Kong AI Gateway Introduction

In April, 2025, Kong announced Kong Gateway 3.10 with the [5th version](https://konghq.com/blog/product-releases/ai-gateway-3-10) of Kong AI Gateway capabilities to address AI-based use cases including automated RAG, PII (Personally Identifiable Information) sanitization and load balance based on tokens and costs.

The diagram below represents the Kong AI Gateway capabilities:

![alt_text](./static/images/kong_ai_gateway.jpg "image_tooltip")


*Kong AI Gateway functional capabilities*

Also, from the architecture perspective, in a nutshell, the Konnect Control Plane and Data Plane nodes topology remains the same.

![alt_text](./static/images/kong_reference_architecture.jpg)


*The Kong AI Gateway sits in between the GenAI applications we build and the LLMs we consume. By leveraging the same underlying core of Kong Gateway, we're reducing complexity in deploying the AI Gateway capabilities as well. And of course, it works on Konnect, Kubernetes, self-hosted, or across multiple clouds.*


## Implementation Architecture

The Agent implementation architecture should include components representing and responsible for the functional scope described above. The architecture comprises:

* Kong AI Gateway to abstract and protect:
    * LLM models
    * external functions used by the AI Agent
* Mistral, Anthropic and OpenAI as LLM model infrastructures
* Redis as the Vector Database
* Ollama as the Embedding Model infrastructure
* Observability layer with Grafana, Loki and Prometheus

Kong AI Gateway, implemented as a regular Konnect Data Plane Node, Redis, Ollama and the Observability layer run on a Minikube Kubernetes Cluster.

![alt_text](./static/images/multi_llm_reference_architecture.jpg)


*Multi-LLM ReAct AI Agent Reference Architecture*

The artifacts used to implement the architecture are available at the following [GitHub repo](https://github.com/CAcquaviva/kong-ai-gateway-langgraph).


## AI Agent Architecture, Reasoning Frameworks and Prompt Engineering

In September, 2024, Google launched a white paper, called [“Agents”](https://www.kaggle.com/whitepaper-agents), exploring the basics of AI Agents, including their architectures and components. The fundamental diagram, included in the paper, is a nice starting point to understand the main moving parts someone working with AI Agents should master.

![alt_text](./static/images/agent.jpg)

*AI Agent components*

The diagram presents three main components of an agent:

* Model: the LLM model that will act as the centralized component responsible for guiding the agent’s decision-making processes.
* Tools: allow Foundation Models to interact with external data and services
* Orchestration: defines a recurring loop in which the agent receives input, plans, reasons and makes decisions to the Agent's next action. Moreover, the Orchestration layer defines “Memories”, used for saving conversations.

An interesting perspective is to think of agents like LLM enhanced with components and capabilities like tools, memory, reasoning, etc.

LangGraph [documentation](https://langchain-ai.github.io/langgraph/agents/agents/) also provides a concise introduction of AI Agents and its main components.


### Reasoning Frameworks

As you can see in the diagram, one of the main responsibilities of the Orchestration layer is controlling the reasoning process implemented by the Model. In fact, at the core of the layer there's a loop responsible for building prompts, making decisions, calling external systems, tracking the steps that have been processed, etc. as it interacts with the Model. The Model, in turn, implements the reasoning process itself supporting a framework.

There are some Reasoning Frameworks defined the LLM models usually support:



* [Chain of Thought](https://arxiv.org/abs/2201.11903) (CoT): based on step-by-step reasoning, defines a linear thought to arrive at an answer. It does not implement tools nor interactions with external components.
* [Tree of Thought](https://arxiv.org/abs/2305.10601) (ToT): extends CoTs with multiple possible solutions creating a tree-like structure, evaluating intermediate states to choose the branch to continue.
* [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning and Action): also based on CoT adding tools, external interaction and observation. Besides, it's supported by Agent Frameworks like LangGraph. Due to all these factors, ReAct has been considered the best Reasoning Framework to implement AI Agent.

In the [blog post](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) the ReAct white paper authors wrote, there's a nice diagram comparing the ReAct Framework with others:


![alt_text](./static/images/react.jpg)



*ReAct Framework comparison*


### Prompt Engineering

Prompt engineering is the practice of crafting inputs (prompts) to get the most useful, accurate, or creative responses from LLM models.

It's critical for AI Agents since you shape how the agents should communicate, think, act, interact with tools, reason and respond. So, the better Prompt Engineering you have the better AI Agent you are going to get.

The same blog post also includes a nice comparison of four prompting methods to respond to a [HotpotQA](https://arxiv.org/abs/1809.09600). HotpotQA is a question answering dataset designed for multi-hop reasoning. You can learn more about it in the [HotpotQA GitHub repository](https://hotpotqa.github.io/).


![alt_text](./static/images/prompt_engineering.jpg)

*Prompt Engineering comparison*


## Simple ReAct based AI Agent

With all these new elements in place, it's time to exercise them. The following code implements a simple ReAct based AI Agent written in Python, including a reasoning loop only, without an Agent Framework or tools interacting with external environments. The AI Agent uses [OpenAI's Chat Completions API](https://platform.openai.com/docs/api-reference/chat?lang=python) to consume the GPT-4 LLM model.

As we have said, the prompt is critical to drive how the Agent should behave. As you can see, ours instructs the Agent with a ReAct pattern including the format it should follow as well as requesting to present the logical steps, with thoughts and observations, used to achieve the final answer.


```
import os
from openai import OpenAI

# Set your OpenAI key
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Initial ReAct-style prompt
def build_prompt(task: str) -> str:
    return f"""
You are an intelligent AI agent solving problems using the ReAct pattern.

Follow this format:
Thought: What are you thinking?
Action: The action you are taking
Observation: (result from imaginary action)
Thought: ...
FINAL_ANSWER: <your final answer>

Task: {task}

Each time you respond, you should only give **one logical step** in your reasoning process.
Do not jump to the final answer immediately.
Each step should be presented using the format above.

Begin.

Thought:
"""

# Reasoning loop with no tools
def run_simple_react_agent(task: str, max_steps: int = 10):
    prompt = build_prompt(task)

    for step in range(max_steps):
        print(f"\n--> Step {step + 1}")


        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        output = response.choices[0].message.content.strip()
        print(output)

        prompt += output + "\n"

        if "FINAL_ANSWER:" in output:
            print("\n Agent finished.")
            break

        # Continue loop by cueing next Thought
        prompt += "Thought:\n"
    else:
        print("\n Max steps reached without a final answer.")

# Run example
run_simple_react_agent("Taking the 'The Grapes of Wrath' novel by John Steinbeck, what are the main differences when comparing it to the movie?")
```


Before executing it, make sure you have installed the `openai` Python package. If you run the script you should get a response like this:


```
--> Step 1
In order to compare the differences between 'The Grapes of Wrath' novel and its movie adaptation, I need to understand the plot, characters, themes, and narrative structure of both the novel and the movie.

Action: 
I will look up summaries, reviews, and analyses of both the novel and the movie to identify key points of similarity and divergence. 

Observation: 
I have gathered information about the plot, characters, themes, and narrative structure of both 'The Grapes of Wrath' novel and its movie adaptation.

--> Step 2
Now that I have the necessary information, I need to compare the two mediums to identify differences. I'll start by comparing the plot of the novel and the movie.

Action:
I will compare the plot of the novel 'The Grapes of Wrath' to the plot of its movie adaptation.

--> Step 3
After comparing the plot of the novel and the movie, I noticed that the movie omits some parts of the novel's plot and simplifies others.

Action: 
Next, I will compare the characters in the novel and the movie to identify any differences.
Observation:
The characters in the movie are generally faithful to the novel, but some have been downplayed or omitted entirely, potentially reducing the complexity and depth of the character relationships and development seen in the novel.

Thought: 
Now, I'll compare the themes presented in the novel and the movie. 

Action: 
I will compare the themes of the novel 'The Grapes of Wrath' to the themes presented in its movie adaptation.

--> Step 4
The themes in the movie are generally consistent with those in the novel, but the film doesn't delve as deeply into some of the more complex and nuanced themes that the novel does.

Action: 
Finally, I will compare the narrative structure of the novel and the movie to identify any differences.

--> Step 5
The narrative structure in the movie is more linear and simplified, whereas the novel uses intercalary chapters to add depth and context to the main story, showing a broader picture of the societal issues of the time.

Action: 
Now, I will compile the differences I have identified into a coherent comparison.

FINAL_ANSWER: 
The main differences between 'The Grapes of Wrath' novel and its movie adaptation are as follows: 
1. Plot: The movie omits and simplifies parts of the novel's plot.
2. Characters: Some characters in the movie are downplayed or omitted, reducing the complexity and depth of the character relationships and development seen in the novel.
3. Themes: The movie doesn't delve as deeply into some of the more complex and nuanced themes that the novel does.
4. Narrative Structure: The movie has a more linear narrative, whereas the novel uses intercalary chapters to add depth and context to the main story.

Agent finished.
```



## Kong Version

As our first exercise, let's inject Kong Data Plane to our simple scenario. Here's the architecture illustrating it:

![alt_text](./static/images/kong_version.jpg)

*AI Agent with Kong Konnect Data Plane*


### OpenAI API support

Kong AI Gateway supports the [OpenAI API specification](https://platform.openai.com/docs/overview) as well as Bedrock and Gemini as incoming formats. The consumer can then send standard OpenAI requests to the Kong AI Gateway. As a basic example, consider this OpenAI request:


```
    curl https://api.openai.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d '{
        "model": "gpt-4o",
        "messages": [
          {
            "role": "user",
            "content": "Hello!"
          }
        ]
      }'
```


When we add Kong AI Gateway, sitting in front of the LLM model, we're not just exposing it but also allowing the consumers to use the same mechanism — in this case, OpenAI APIs — to consume it. That leads to a very flexible and powerful capability when we come to development processes. In other words, Kong AI Gateway normalizes the consumption of any LLM infrastructure, including Amazon Bedrock, Mistral, OpenAI, Cohere, etc.

As an exercise, the new request should be something like this. The request has some minor differences:



* It sends a request to the Kong API Gateway Data Plane Node.
* It replaces the OpenAI endpoint with a Kong API Gateway route.
* The API Key is actually managed by the Kong API Gateway now.
* It does not refer to any model, since it's being resolved by the AI Gateway.

    ```
    curl http://$DATA_PLANE_LB/bedrock-route \
      -H "Content-Type: application/json" \
      -H "apikey: $KONG_API_KEY" \
      -d '{
         "messages": [
           {
             "role": "user",
             "content": "Hello!"
           }
         ]
       }'

    ```



### Minikube installation

We are going to deploy our Data Plane in a [Minikube](https://minikube.sigs.k8s.io/) Cluster over Docker or Podman. For example, you can start Podman with:


```
podman machine set --memory 8196
podman machine start
```


If you want to stop it run:


```
podman machine stop
```


Then you can install Minikube with:


```
minikube start --driver=podman --memory='no-limit'
```


To be able to consume the Kubernetes Load Balancer Services, in another terminal run:


```
minikube tunnel
```



### Konnect subscription

Now you have to get access to Konnect. Click on the [Registration](https://konghq.com/products/kong-konnect/register) link and present your credentials. Or, if you already have a Konnect subscription, [log in to it](https://cloud.konghq.com/us).


### Konnect PAT

In order to interact with Konnect, using command lines, you need a [Konnect Personal Access Token (PAT)](https://docs.konghq.com/konnect/gateway-manager/declarative-config/#generate-a-personal-access-token). To generate your PAT, go to Konnect UI, click on your initials in the upper right corner of the Konnect home page, then select "Personal Access Tokens." Click on "+ Generate Token," name your PAT, set its expiration time, and be sure to copy and save it as an environment variable also named as PAT. Konnect won’t display your PAT again.


### Kong Gateway Operator (KGO), Konnect Control Plane creation and Data Plane deployment

For Kubernetes deployments, Kong provides [Kong Gateway Operator (KGO)](https://docs.konghq.com/gateway-operator), an Operator capable of managing all flavours of Kong installations, including Kong Ingress Controller, Kong Gateway Data Planes, for self-managed or Konnect based deployments.

Our topology comprises a hybrid deployment where a Data Plane, running on Minikube, is connected to the Konnect Control Plane.

Start adding the [KGO Helm Charts](https://github.com/Kong/charts/tree/main/charts/gateway-operator) to your environment:


```
helm repo add kong https://charts.konghq.com
helm repo update kong
```


Install the Operator with:


```
helm upgrade --install kgo kong/gateway-operator \
  -n kong-system \
  --create-namespace \
  --set image.tag=1.6.1 \
  --set kubernetes-configuration-crds.enabled=true \
  --set env.ENABLE_CONTROLLER_KONNECT=true
```


You can check the Operator's logs with:


```
kubectl logs -f $(kubectl get pod -n kong-system -o json | jq -r '.items[].metadata | select(.name | startswith("kgo-gateway"))' | jq -r '.name') -n kong-system
```


And if you want to uninstall it run:


```
helm uninstall kgo -n kong-system
kubectl delete namespace kong-system
```



#### Konnect Control Plane creation

The first thing to do, in order to get your Konnect Control Plane defined, you have to create a Kubernetes Secret with your PAT. KGO requires your secret to be labeled. The commands should be like these ones: 


```
kubectl create namespace kong

kubectl delete secret konnect-pat -n kong
kubectl create secret generic konnect-pat -n kong --from-literal=token='kpat_K6Cgbx…..'

kubectl label secret konnect-pat -n kong "konghq.com/credential=konnect"
```


Then, the following declaration defines an [Authentication Configuration](https://docs.konghq.com/gateway-operator/latest/reference/custom-resources/#konnectapiauthconfiguration), based on the Kubernetes Secret and referring to a Konnect API URL, and the actual [Konnect Control Plane](https://docs.konghq.com/gateway-operator/1.5.x/reference/custom-resources/#konnectgatewaycontrolplane). Check the [documentation](https://docs.konghq.com/gateway-operator/latest/get-started/konnect/create-konnectextension/#create-an-access-token-in-konnect) to learn more about it.


```
cat <<EOF | kubectl apply -f -
kind: KonnectAPIAuthConfiguration
apiVersion: konnect.konghq.com/v1alpha1
metadata:
  name: konnect-api-auth-conf
  namespace: kong
spec:
  type: secretRef
  secretRef:
    name: konnect-pat
    namespace: kong
  serverURL: us.api.konghq.com
---
kind: KonnectGatewayControlPlane
apiVersion: konnect.konghq.com/v1alpha1
metadata:
 name: ai-gateway
 namespace: kong
spec:
 name: ai-gateway
 konnect:
   authRef:
     name: konnect-api-auth-conf
EOF
```


You should see your Control Plane listed in your Konnect Organization:



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")


*Kong Konnect Control Plane*


#### Konnect Data Plane deployment

The next declaration instantiates a Data Plane connected to your Control Plane. It creates a [KonnectExtension](https://docs.konghq.com/gateway-operator/1.5.x/reference/custom-resources/#konnectextension-1), asking KGO to manage the certificate and private key provisioning automatically, and the actual Data Plane. The [Data Plane](https://docs.konghq.com/gateway-operator/latest/reference/custom-resources/#dataplane) declaration specifies the Docker image, in our case 3.10, as well as how the Kubernetes Service, related to the Data Plane, should be created.

The Data Plane declaration also defines the “[unstrusted_lua_sandbox_requires](https://docs.konghq.com/gateway/latest/reference/configuration/#untrusted_lua_sandbox_requires)” environment variables with functions used by the Observability layer in order to allow them to be executed inside the Data Plane's Lua sandbox.


```
cat <<EOF | kubectl apply -f -
kind: KonnectExtension
apiVersion: konnect.konghq.com/v1alpha1
metadata:
 name: konnect-config1
 namespace: kong
spec:
 clientAuth:
   certificateSecret:
     provisioning: Automatic
 konnect:
   controlPlane:
     ref:
       type: konnectNamespacedRef
       konnectNamespacedRef:
         name: ai-gateway
---
apiVersion: gateway-operator.konghq.com/v1beta1
kind: DataPlane
metadata:
 name: dataplane1
 namespace: kong
spec:
 extensions:
 - kind: KonnectExtension
   name: konnect-config1
   group: konnect.konghq.com
 deployment:
   podTemplateSpec:
     spec:
       containers:
       - name: proxy
         image: kong/kong-gateway:3.10
         env:
         - name: KONG_UNTRUSTED_LUA_SANDBOX_REQUIRES
           value: pl.stringio, ffi-zlib, cjson.safe
 network:
   services:
     ingress:
       name: proxy1
       type: LoadBalancer
EOF
```


You can check the Data Plane logs with


```
kubectl logs -f $(kubectl get pod -n kong -o json | jq -r '.items[].metadata | select(.name | startswith("dataplane-"))' | jq -r '.name') -n kong
```


Also, you should see your first Data Plane listed and connected to the previously created Konnect Control Plane:



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")


*Kong Konnect Data Plane*


#### Consume the Data Plane

If you check the new Kubernetes Service, since it's been created as “Load Balancer”, and we are running Minikube, you'll see that its external ip is defined as “127.0.0.1”.


```
% kubectl get service proxy1 -n kong
NAME     TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)                      AGE
proxy1   LoadBalancer   10.103.170.14   127.0.0.1     80:30350/TCP,443:30510/TCP   7m34s
```


You should also see that, in the terminal running Minikube's tunnel, the “proxy1” service has been exposed. It might need to type your admin's password.


```
❗  The service/ingress proxy1 requires privileged ports to be exposed: [80 443]
🔑  sudo permission will be asked for it.
🏃  Starting tunnel for service proxy1.
```


If you send a request to it you should get a response from the Data Plane


```
% curl -i http://localhost
HTTP/1.1 404 Not Found
Date: Thu, 24 Apr 2025 14:07:26 GMT
Content-Type: application/json; charset=utf-8
Connection: keep-alive
Content-Length: 103
X-Kong-Response-Latency: 1
Server: kong/3.10.0.1-enterprise-edition
X-Kong-Request-Id: 2b83f976af69655067bb5a4320a4677f

{
  "message":"no Route matched with those values",
  "request_id":"2b83f976af69655067bb5a4320a4677f"
}
```


Finally, if you want to delete all components run:


```
kubectl delete dataplane dataplane1 -n kong
kubectl delete konnectextensions.konnect.konghq.com konnect-config1 -n kong
kubectl delete konnectgatewaycontrolplane ai-gateway -n kong
kubectl delete konnectapiauthconfiguration konnect-api-auth-conf -n kong
kubectl delete secret konnect-pat -n kong
kubectl delete namespace kong
```



### Creating Kong Objects using decK

In this next step, you'll create the Kong Objects required to consume the OpenAI LLM model: Kong Service, Kong Route and Kong Plugin. You can continue using KGO and Kubernetes style declarations. However, we are going to exercise [decK](https://docs.konghq.com/deck/latest/) (declarations for Kong). With decK you can manage Kong Konnect configuration and create Kong Objects in a declarative way. Check the decK documentation to learn how to [install](https://docs.konghq.com/deck/latest/installation/) it.

Once you have decK installed, you should ping Konnect to check if the connection is up using the same PAT you created previously. You can use the following command to do it:


```
% deck gateway ping --konnect-token $PAT
Successfully Konnected to the Kong organization!
```



### decK declaration

Now, create a file named "kong_agent_simple.yaml” with the following decK declaration:


```
cat > kong_agent_simple.yaml << 'EOF'
_format_version: "3.0"
_info:
  select_tags:
  - agent
_konnect:
  control_plane_name: ai-gateway
services:
- name: agent-service
  host: localhost
  port: 32000
  routes:
  - name: agent-route1
    paths:
    - /agent-route
    plugins:
    - name: ai-proxy-advanced
      instance_name: "ai-proxy-advanced-openai-agent"
      enabled: true
      config:
        targets:
        - auth:
            header_name: "Authorization"
            header_value: "Bearer <your_OPENAI_API_KEY>"
          route_type: "llm/v1/chat"
          model:
            provider: "openai"
            name: "gpt-4"
EOF
```


The declaration defines multiple Kong Objects:



* [Kong Gateway Service](https://docs.konghq.com/gateway/latest/get-started/services-and-routes/) named "agent-service". The service doesn’t need to map to any real upstream URL. In fact, it can point somewhere empty, for example, http://localhost:32000. This is because the AI Proxy plugin, also configured in the declaration, overwrites the upstream URL. This requirement will be removed in a later Kong revision.
* Kong Route: the Gateway Service has a route defined with the "/agent-route" path. That's the route we're going to consume to reach OpenAI's GPT-4 LLM.
* AI Proxy Advanced Plugin. It's configured to consume OpenAI's “gpt-4” model. The “route_type” parameter, set as “llm/v1/chat”, refers to OpenAI's “https://api.openai.com/v1/chat/completions” endpoint. Kong recommends storing the API Keys as secrets in a [Secret Manager](https://docs.konghq.com/gateway/latest/kong-enterprise/secrets-management/) like AWS Secrets Manager or HashiCorp Vault. The current configuration, including the OpenAI API Key in the declaration, is for lab environments only, not recommended for production. Please refer to the official [AI Proxy Advanced Plugin documentation page](https://docs.konghq.com/hub/kong-inc/ai-proxy-advanced/) to learn more about its configuration.

The declaration has been tagged as "agent" so you can manage its objects without impacting any other ones you might have created previously. Also, note the declaration is saying it should be applied to your "ai-gateway" Konnect Control Plane.

You can submit the declaration with the following decK command:


```
deck gateway sync --konnect-token $PAT kong_agent_simple.yaml
```


If you want to destroy all objects run:


```
deck gateway reset --konnect-control-plane-name "ai-gateway" --select-tag "agent" --konnect-token $PAT -f
```



### AI Agent

Now, we're ready to get back to our Agent. There are some changes you have to apply to your code to make Kong Version out of it:



* To redirect the requests to Kong AI Data Plane, you have to change your OpenAI constructor to refer to Kong Route.
* Note that the OpenAI constructor still requires the OpenAI API Key. However, since it's managed by Kong, you can use any value for it.
* Since the AI Proxy Advanced Plugin has been configured to consume the GPT-4 model, you can set the “model” parameter you have in the “chat.completions.create” call as an empty string.
*  And, of course, you can delete the line where you import the “os” package.

Here's the new constructor:


```
…
client = OpenAI(
    base_url="http://localhost:80/agent-route",
    api_key="dummy"
)
…
```


If you run your new code, you are supposed to get similar results, as Kong AI Gateway is not applying any policy to your requests. So far, if you check your Data Plane, you'll see all requests that have been processed by the Data Plane and routed to OpenAI's GPT-4 model.

Here’s the Kong Service Analytics tab:



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")


*Kong Konnect Analytics*

As you can see, the Agent sent 5 requests to Kong Data Plane (and therefore, GPT-4), one for each step it has processed to respond to our question.

Moreover, the Konnect Control Plane provides extensive Analytics capabilities where you can check all requests processed by the Data Planes. Click on “Analytics” menu option and “Requests”:



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")


*Kong Konnect processed requests*

Note you can introspect all processed requests including all major data like HTTP method, Kong Route and Service, Upstream URI, etc. For performance and security reasons, the Data Plane does not report the Control Plane with the body of the requests, which might be interesting to check, especially for situations like AI Agents. In the next section, we are going to configure the Data Plane to externalize both requests and responses bodies


### Kong AI Gateway Plugins

Before we jump to the next section, it's important to keep in mind that Kong AI Gateway provides an extensive list of AI-related plugins to implement specific policies related to:



* Prompt Engineering
* Semantic Processing
* Rate Limiting based on tokens
* Request and Response transformations based on LLM queries
* PII (Personally Identifiable Information) sanitization

As we stated before, this blog post will leverage Semantic Routing capabilities to manage multiple LLMs sitting behind and getting protected by the Kong AI Gateway.

That concludes part I of the series. In part II we are going to explore the fundamentals of LangGraph to create an AI Agent including Kong AI Gateway, Tools and Function Calling.
