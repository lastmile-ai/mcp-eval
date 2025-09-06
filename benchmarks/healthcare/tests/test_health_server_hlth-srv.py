import pytest
import mcp_eval
from mcp_eval import Expect
from mcp_eval.session import TestAgent
from mcp_agent.agents.agent_spec import AgentSpec

# Pin tests to the intended server by configuring a suite-level AgentSpec.
# This avoids relying on whatever the current default agent is in mcpeval.yaml.
@mcp_eval.setup
def _configure_suite_agent():
    mcp_eval.use_agent(
        AgentSpec(
            name="generated-pytest",
            instruction="You are a helpful assistant that can use MCP servers effectively.",
            server_names=["health_server"],
        )
    )

@pytest.mark.asyncio
async def test_basic_drug_information_lookup(agent: TestAgent):
    response = await agent.generate_str("Tell me about the drug aspirin - what is it used for and any basic information")
    await agent.session.assert_that(Expect.tools.was_called("fda_drug_lookup", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fda_drug_lookup", {'drug_name': 'aspirin', 'search_type': 'general'}))
    await agent.session.assert_that(Expect.judge.llm("Response provides comprehensive information about aspirin including its uses and basic drug information", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("fda_drug_lookup", {'drug_name': 'aspirin'}))
    await agent.session.assert_that(Expect.content.contains("aspirin", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("pain", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("fever", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("inflammation", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(10000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide accurate basic information about aspirin including its primary uses (pain relief, fever reduction, anti-inflammatory effects) and be informative for a general audience seeking drug information.", min_score=0.7), response=response)

@pytest.mark.asyncio
async def test_drug_adverse_events_lookup(agent: TestAgent):
    response = await agent.generate_str("I\u0027m concerned about side effects of metformin. Can you look up reported adverse events?")
    await agent.session.assert_that(Expect.tools.was_called("fda_drug_lookup", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fda_drug_lookup", {'drug_name': 'metformin', 'search_type': 'adverse_events'}))
    await agent.session.assert_that(Expect.content.contains("adverse", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response addresses the user\u0027s concern about metformin side effects with appropriate adverse event information", min_score=0.75), response=response)
    await agent.session.assert_that(Expect.content.contains("metformin", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("side effect", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(3))
    await agent.session.assert_that(Expect.performance.response_time_under(12000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide specific information about metformin adverse events from FDA data, address the user\u0027s concern about side effects, and present the information in a clear, helpful manner without providing medical advice.", min_score=0.75), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["fda_drug_lookup"], allow_other_calls=True))

@pytest.mark.asyncio
async def test_pubmed_literature_search(agent: TestAgent):
    response = await agent.generate_str("Find recent research papers about COVID-19 vaccines from the last 3 years. I need up to 10 results.")
    await agent.session.assert_that(Expect.tools.was_called("pubmed_search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("pubmed_search", {'query': 'COVID-19 vaccines', 'max_results': 10, 'date_range': '3'}))
    await agent.session.assert_that(Expect.content.contains("research", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides relevant COVID-19 vaccine research papers from recent years", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("COVID-19", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("vaccine", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("PubMed", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(15000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should present recent COVID-19 vaccine research papers from PubMed, include relevant titles and publication information, be limited to the last 3 years, and provide up to 10 results as requested.", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["pubmed_search"], allow_other_calls=False))

@pytest.mark.asyncio
async def test_health_topic_diabetes_spanish(agent: TestAgent):
    response = await agent.generate_str("I need information about diabetes in Spanish for a patient education material")
    await agent.session.assert_that(Expect.tools.was_called("health_topics_search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("health_topics_search", {'topic': 'diabetes', 'language': 'es'}))
    await agent.session.assert_that(Expect.content.contains("diabetes", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides diabetes information suitable for Spanish-speaking patient education", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("Spanish", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("patient education", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(8000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide diabetes health information suitable for Spanish-speaking patients, acknowledge the language request, and present evidence-based content appropriate for patient education materials.", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["health_topics_search"], allow_other_calls=False))

@pytest.mark.asyncio
async def test_clinical_trials_cancer_recruiting(agent: TestAgent):
    response = await agent.generate_str("Help me find currently recruiting clinical trials for breast cancer patients. Show me 15 results.")
    await agent.session.assert_that(Expect.tools.was_called("clinical_trials_search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("clinical_trials_search", {'condition': 'breast cancer', 'status': 'recruiting', 'max_results': 15}))
    await agent.session.assert_that(Expect.content.contains("recruiting", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides relevant recruiting clinical trials for breast cancer with appropriate details", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.content.contains("breast cancer", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("clinical trial", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("15", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(12000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide information about currently recruiting clinical trials for breast cancer patients, present up to 15 results as requested, include relevant trial details, and clearly indicate the recruiting status.", min_score=0.85), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["clinical_trials_search"], allow_other_calls=False))

@pytest.mark.asyncio
async def test_icd10_code_lookup(agent: TestAgent):
    response = await agent.generate_str("What is the ICD-10 code for hypertension? Look it up for me.")
    await agent.session.assert_that(Expect.tools.was_called("medical_terminology_lookup", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("medical_terminology_lookup", {'description': 'hypertension', 'max_results': 10}))
    await agent.session.assert_that(Expect.content.contains("ICD-10", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response provides accurate ICD-10 code information for hypertension", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("medical_terminology_lookup", {'description': 'hypertension'}))
    await agent.session.assert_that(Expect.content.contains("hypertension", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("I10", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(6000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide the correct ICD-10 code for hypertension (I10), clearly identify it as the requested code, and demonstrate successful use of the medical terminology lookup tool.", min_score=0.9), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["medical_terminology_lookup"], allow_other_calls=False))

@pytest.mark.asyncio
async def test_comprehensive_drug_research(agent: TestAgent):
    response = await agent.generate_str("I need comprehensive information about ibuprofen including FDA data, recent research, and any clinical trials. Please search multiple sources.")
    await agent.session.assert_that(Expect.tools.was_called("fda_drug_lookup", min_times=1))
    await agent.session.assert_that(Expect.tools.was_called("pubmed_search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fda_drug_lookup", {'drug_name': 'ibuprofen', 'search_type': 'general'}))
    await agent.session.assert_that(Expect.tools.called_with("pubmed_search", {'query': 'ibuprofen', 'max_results': 5}))
    await agent.session.assert_that(Expect.judge.llm("Response synthesizes information from multiple sources to provide comprehensive ibuprofen information", min_score=0.85), response=response)
    await agent.session.assert_that(Expect.tools.was_called("clinical_trials_search", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("fda_drug_lookup", {'drug_name': 'ibuprofen'}))
    await agent.session.assert_that(Expect.tools.called_with("pubmed_search", {'query': 'ibuprofen'}))
    await agent.session.assert_that(Expect.tools.called_with("clinical_trials_search", {'condition': 'ibuprofen'}))
    await agent.session.assert_that(Expect.content.contains("ibuprofen", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("FDA", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("research", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("clinical trial", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("comprehensive", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(5))
    await agent.session.assert_that(Expect.performance.response_time_under(20000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should provide comprehensive information about ibuprofen from multiple sources including FDA data, recent research from PubMed, and clinical trial information. It should demonstrate use of multiple tools and synthesize information effectively.", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["fda_drug_lookup", "pubmed_search", "clinical_trials_search"], allow_other_calls=True))

@pytest.mark.asyncio
async def test_icd10_reverse_lookup(agent: TestAgent):
    response = await agent.generate_str("I have ICD-10 code I10. Can you tell me what medical condition this represents?")
    await agent.session.assert_that(Expect.tools.was_called("medical_terminology_lookup", min_times=1))
    await agent.session.assert_that(Expect.tools.called_with("medical_terminology_lookup", {'code': 'I10', 'max_results': 10}))
    await agent.session.assert_that(Expect.content.contains("I10", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.judge.llm("Response correctly identifies the medical condition associated with ICD-10 code I10", min_score=0.8), response=response)
    await agent.session.assert_that(Expect.tools.called_with("medical_terminology_lookup", {'code': 'I10'}))
    await agent.session.assert_that(Expect.content.contains("hypertension", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("essential hypertension", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.content.contains("medical condition", case_sensitive=False), response=response)
    await agent.session.assert_that(Expect.performance.max_iterations(2))
    await agent.session.assert_that(Expect.performance.response_time_under(5000.0))
    await agent.session.assert_that(Expect.judge.llm("The response should correctly identify that ICD-10 code I10 represents essential hypertension, provide clear explanation of what this medical condition is, and demonstrate successful reverse lookup of the code.", min_score=0.9), response=response)
    await agent.session.assert_that(Expect.tools.sequence(["medical_terminology_lookup"], allow_other_calls=False))


