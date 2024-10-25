import requests
import json
from models.gpt import *
from bs4 import BeautifulSoup
import html2text
import json
import os
import html2text
import tiktoken
import logging

from rag.retrieve import find_relative_chunk
enc = tiktoken.get_encoding("cl100k_base")
class Study:

    def __init__(self, abstract_url,drug_name,compressor=None,initial_data=None):
        self.compressor = compressor
        if initial_data:
            self.data = initial_data
        else:
            self.data = {
                "base_info":{
                    "abstract_url": abstract_url,
                    "drug_name": drug_name
                },
                "paper_info":{
                "title": "",
                "year": "",
                "doi":"",
                "abstract": "",
                "content":"",
                "includes_pediatrics": False,
                "proves_effective": False,
                "age": "",
                "disease": "",
                "icd10":"",
                "drug_route": "",
                "study_type": ""    
            },
            "state":{
                "need_to_fetch_conent": False,
                "no_abstract": False,
                "include_abstract": False,
                "include_content": False,
                "process_pediatrics_inabstract": False,
                "process_effectiveness": False,
                "process_pediatrics_incontent": False,
                "process_population_effectiveness": False,
                "remain_problem": False,
                "finished":False
            },
            "answer":{
                "process_pediatrics_inabstract":"",
                "process_effectiveness":"",
                "process_pediatrics_incontent":"",
                "process_population_effectiveness":"",
                "age":"",
                "target_disease":"",
                "icd10_of_target_disease":"",
                "drug_intervention":"",
                "study_type":""
            }
            }



    def fetch_abstract(self,abstract=None,title=None,year=None):
        if self.data["paper_info"]["abstract"] != "":
            return True
        if abstract!=None:
            if len(abstract)<10:
                logging.error(f"name:{self.data['base_info']['drug_name']},pmid:{self.data['base_info']['abstract_url']} no abstract")
                self.data["state"]["no_abstract"] = True
                self.data["state"]["need_to_fetch_conent"] = True
                return False
            self.data["paper_info"]["abstract"] = abstract
            self.data["paper_info"]["title"] = title
            self.data["paper_info"]["year"] = year
            self.data["state"]["include_abstract"] = True
            return True
        url = 'https://pubmed.ncbi.nlm.nih.gov/' + self.data["base_info"]["abstract_url"] + "/" 
        try:
            response = requests.get(url)
            if response.status_code == 200:
            # 解析HTML
                soup = BeautifulSoup(response.text, 'html.parser') 
                abstract_div = soup.find('div', class_='abstract-content selected', id='eng-abstract')
                time_span = soup.find('span', class_="cit")
                title_h1 = soup.find('h1', class_="heading-title")
                doi_a = soup.find('a', class_="id-link")
                if time_span is None:
                    html_time = ""
                else:
                    html_time = time_span.get_text()
                if title_h1 is None:
                    html_title = ""
                else:
                    html_title = title_h1.get_text()
                if doi_a is None:
                    html_doi = ""
                else:
                    html_doi = doi_a.get_text()
                time = html2text.html2text(html_time)
                title = html2text.html2text(html_title)
                doi = html2text.html2text(html_doi)  
                if abstract_div is not None:
                    html_abstract = abstract_div.get_text()
                else:
                    logging.error(f"drug name:{self.data['base_info']['drug_name']}'s paper{self.data['base_info']['abstract_url']} not have abstract")
                    return False
                markdown = html2text.html2text(html_abstract)  
                if len(markdown)<10:
                    logging.error(f"name:{self.data['base_info']['drug_name']},pmid:{self.data['base_info']['abstract_url']} no abstract")
                    self.data["state"]["no_abstract"] = True
                    self.data["state"]["need_to_fetch_conent"] = True
                else:
                    self.data["paper_info"]["abstract"] = markdown
                    self.data["state"]["include_abstract"] = True
                self.data["paper_info"]["title"] = title
                self.data["paper_info"]["year"] = time
                self.data["paper_info"]["doi"] = doi
                
                logging.info(f"Abstract: {markdown} have been fetched.")
                return True
        except requests.RequestException as e:
            logging.error(f"Error fetching abstract: {e}")
            return False
    
    def fetch_content(self,content=None):
        if content!=None:
            self.data["content"] = content
            self.state["include_content"] = True
            return True
        """"""



    def process_pediatrics_inabstract(self):
        if self.data["state"]["process_pediatrics_inabstract"] == True:
            json_object = self.data["answer"]["process_pediatrics_inabstract"]
            return json_object
        json_object = {"reason": "","short_answer": "[final model answer (e.g. included,excluded,age_not_mentioned]"}
        prompt = "##prompt:Use the criteria below to inform your decision. If any exclusion criteria are met or not all inclusion criteria are met, exclude the article, lf all inclusion criteria are met, include the article. Please type \"included\"or \"excluded\" to indicate your decision. ##Title:{title}\n##Abstract:{abstract}\n\n##Inclusion criteria:\nDrug Efficacy Study in human\nAt least one pediatric(<18 years) in sample patients.\n\n##Attention:If no age information is mentioned in the abstract,return \"age_not_mentioned\"\n##model generated chain of thought explanation.\nTherefore, the json format answer is \n{json}"
        prompt = prompt.format(abstract=self.data["paper_info"]["abstract"],title = self.data["paper_info"]["title"],json=json_object)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["process_pediatrics_inabstract"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None

        

    def process_effectiveness(self):
        if self.data["state"]["process_effectiveness"] == True:
            json_object = self.data["answer"]["process_effectiveness"]
            return json_object
        json_object = {"reason": "","short_answer": "[final model answer (e.g. Yes,No,NA,NM)]"}
        prompt = "##question:Only using the information provided in \"Text for Analysis\", assess whether the target drug {target_drug}, shows significant efficacy or effectiveness (including improvement in survival) in pediatric populations.\nLet's think step by step. Follow these guidelines:\n**Control group: Identify whether this study divided patients into two or more groups receiving different intervention regimens (also known as treatment regimens), and whether this study included a control group or more.\nIs target drug involved: Examine the intervention regimens across all groups in the study, identify which regimens in this study involve the target drug {target_drug}. This includes groups receiving the drug as monotherapy, as part of combination therapy with other medications, or as an active ingredient in a regimen that also includes a placebo.\nEffective: If the text shows that the regimen including {target_drug} has significant efficacy or effectiveness (including improvement in survival or other clinical outcome measures) compared to a control group that does not involving {target_drug} in pediatric patients, state that the regimens including this drug is more effective than other regimens that do not include this drug in this population.\nNA (Not Applicable): If the study is a single-arm study or does not include a control group, or if all intervention regimens include the target drug {target_drug}, respond with \"NA\".\nNM (Not Mentioned): If the text does not mention {target_drug}, or if the drug is not the subject of the study, or if the study is not conducted on pediatric populations, respond with \"NM\".\nPlease return \"Yes\", \"No\", \"NA (Not Applicable)\", or \"NM (Not Mentioned)\", and provide your assessment based on the provided text and explain your reasoning.\n##Text for Analysis:\n##Title:{title}\n##Abstract:{abstract}\n\n##Your Assessment:\nmodel generated chain of thought explanation\nTherefore, the json format answer is \n{json}"
        prompt = prompt.format(abstract=self.data["paper_info"]["abstract"],title = self.data["paper_info"]["title"],json=json_object,target_drug=self.data["base_info"]["drug_name"])
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["process_effectiveness"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None

    def conmpress_content(self,relative_chunk,question):
        content = []
        all_token = 0
        for key, value in relative_chunk.items():
            content.append(value)
            all_token =  all_token + len(enc.encode(value) )
        # contents = '\n'.join(content) 
        if all_token >2000:
            print("all_token lenght:" + str(all_token))
            contents = self.compressor.compress_prompt(
                content,
                question = question,
                rate=0.6,
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3, # or 0.4
                condition_compare=True,
                context_budget="+100"
            )
        else:
            contents = '\n'.join(content)
            return contents
            
        return contents["compressed_prompt"]

 

    def process_pediatrics_incontent(self):
        if self.data["state"]["process_pediatrics_incontent"] == True:
            json_object = self.answer["process_pediatrics_incontent"]
            return json_object
        json_object = {"reason":"","short_answer":""}
        question = "Is the pediatric (<18 years) population included in this study?"
        if self.data["state"]["include_content"] == False:
            return None
        relative_chunk = find_relative_chunk(self.data["paper_info"]["content"],question)
        if relative_chunk == None:
            return None
        contents = self.conmpress_content(relative_chunk,question)
        prompt = "##INSTRUCTION:According to the relative information, is the pediatric (<18 years) population included in this study?your answer must is in [yes,no]\n##Relative information:{context}\n##model generated chain of thought explanation.\nTherefore, the json format answer is json:{json}"
        prompt = prompt.format(context= contents,json=json_object)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["process_pediatrics_incontent"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None
        # json_obj = {'reason': 'The study specifically focused on preterm infants, categorizing them by gestational age and birth weight. Since pediatric population generally refers to children from birth to adolescence, and in this context, all subjects are preterm infants (which are newborns), it is correct to interpret that this pediatric subset was included in the study.', 'short_answer': 'yes'}
        

    def process_population_effectiveness(self):
        if self.data["state"]["process_population_effectiveness"] == True:
            json_object = self.data["answer"]["process_population_effectiveness"]
            return json_object
        json_object = {"reason":"","short_answer":""}
        prompt = "##question:Only using the information provided in \"Text for Analysis\", assess whether the target drug {target_drug}, shows significant efficacy or effectiveness (including improvement in survival) in populations.\nLet's think step by step. Follow these guidelines:\n**Control group: Identify whether this study divided patients into two or more groups receiving different intervention regimens (also known as treatment regimens), and whether this study included a control group or more.\nIs target drug involved: Examine the intervention regimens across all groups in the study, identify which regimens in this study involve the target drug {target_drug}. This includes groups receiving the drug as monotherapy, as part of combination therapy with other medications, or as an active ingredient in a regimen that also includes a placebo.\nEffective: If the text shows that the regimen including {target_drug} has significant efficacy or effectiveness (including improvement in survival or other clinical outcome measures) compared to a control group that does not involving {target_drug} in  patients, state that the regimens including this drug is more effective than other regimens that do not include this drug in this population.\nNA (Not Applicable): If the study is a single-arm study or does not include a control group, or if all intervention regimens include the target drug {target_drug}, respond with \"NA\".\nNM (Not Mentioned): If the text does not mention {target_drug}, or if the drug is not the subject of the study, or if the study is not conducted on populations, respond with \"NM\".\nPlease return \"Yes\", \"No\", \"NA (Not Applicable)\", or \"NM (Not Mentioned)\", and provide your assessment based on the provided text and explain your reasoning.\n##Text for Analysis:##Title:{title}\n##Abstract:{abstract}\n\n##Your Assessment:\nmodel generated chain of thought explanation\nTherefore, the json format answer is \n{json}"
        prompt = prompt.format(target_drug = self.data["base_info"]["drug_name"],abstract=self.data["paper_info"]["abstract"],title = self.data["paper_info"]["title"],json=json_object)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["process_population_effectiveness"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None
        

    def ask_remain_question(self):
        self.data["paper_info"]["age"] = self.age()
        self.data["paper_info"]["disease"] = self.target_disease()
        self.data["paper_info"]["icd10"] = self.icd10_of_target_disease()
        self.data["paper_info"]["drug_route"] = self.drug_intervention()
        self.data["paper_info"]["study_type"] = self.study_type()
        self.data["state"]["remain_problem"] = True
        self.data["state"]["finished"] = True

    def age(self):
        template = """
##Question:{question}
##Context:\"""{context}\"""
##Please think step by step and generate the final answer. You need to provide the answer and reasons in standard JSON format.
{json}
"""
        prompt = ""
        js = """{
        "pepople_age_range": "",
        "Premature_infants_isin": "",
        "Newborns_isin": "",
        "Infants_isin": "",
        "Children_isin": "",
        "Adolescents_isin": ""
      }"""
        question = "What is the age distribution of the people in the study? \nExtract data on the age distribution of the participants.\nIf available, note any provided statistics on average age, range, or age groups.\nSummarize the size of the experimental population.\nUse the age distribution to answer whether the following groups are involved, and return Yes or No in Premature_infants_isin and other similar fields (\npremature baby\nNewborn <28 days\nInfants and toddlers 28 days to 23 months\nChildren 2-11 years old\nAdolescents 12-17 years old\n)?"
        relative_chunk = find_relative_chunk(self.data["paper_info"]["content"],question)
        if relative_chunk == None:
            return None
        contents = self.conmpress_content(relative_chunk,question)
        prompt = template.format(question = question,context = contents,json = js)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["age"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None
        


    def target_disease(self):
        template = """
##Question:{question}
##Context:\"""{context}\"""
##Please think step by step and generate the final answer. You need to provide the answer and reasons in standard JSON format.
{json}
"""
        js =  """{
            "reason": "",
            "disease": ""
        }"""
        prompt = ""
        question = "What the main disease targeted in this study?"
        relative_chunk = find_relative_chunk(self.data["paper_info"]["content"],question)
        if relative_chunk == None:
            return None
        contents = self.conmpress_content(relative_chunk,question)
        prompt = template.format(question=question, context=contents, json=js)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["target_disease"] = json_object
            self.data["paper_info"]["disease"] = json_object["disease"]
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None


    def icd10_of_target_disease(self):
        if self.data["paper_info"]["disease"] == "":
            self.data["answer"]["disease"] = self.target_disease()
            disease =  self.data["disease"]
        else:
            disease =  self.data["paper_info"]["disease"]
        result = get_request_data_icd10(disease)
        try:
            json_object = json.loads(result)
            self.data["paper_info"]["icd10"] = json_object
            self.data["answer"]["icd10_of_target_disease"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None

    def drug_intervention(self):
        template = """
##Question:{question}
##Context:\"""{context}\"""
##Please think step by step and generate the final answer. You need to provide the answer and reasons in standard JSON format.
{json}
"""     
        js  ="""{
            "reason": "",    
            "route_of_administration": ""
            
        }"""
        question = f"What is the route of administration of the target drug {self.data['base_info']['drug_name']}"
        relative_chunk = find_relative_chunk(self.data["paper_info"]["content"],question)
        if relative_chunk == None:
            return None
        contents = self.conmpress_content(relative_chunk,question)
        prompt = template.format(question=question, context=contents, json=js)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["drug_intervention"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None

    def study_type(self):
        template = """
##Question:{question}
##Context:\"""{context}\"""
##Please think step by step and generate the final answer. You need to provide the answer and reasons in standard JSON format.
{json}
""" 
        question = "what type of literature does this study belong to (Please select one answer from the following: Systematic review, RCT, Non-randomized controlled cohort study, Non-randomized controlled follow-up study,retrospective cohort study, Case report Case-series, Case-control study Historically controlled study, Single-arm study, Narrative literature review)? Based on the following fields, determine whether it is a systematic review/meta-analysis, whether it includes RCT, whether it includes a control group or multiple control groups, and whether this study used cohort or follow-up data?. For questions such as \"is_Meta_analysis_or_systematic_review\", please answer only Yes or No in similar fields such as \"is_Meta_analysis_or_systematic_review\"?"
        js = """{
                "reason": "",
                "type": "",
                "is_Meta_analysis_or_systematic_review": "",
                "RCT_involved": "",
                "control_group_or_multiple_intervention_groups_involved": "",
                "used_cohort_or_follow-up_data": ""
            }"""
        relative_chunk = find_relative_chunk(self.data["paper_info"]["content"],question)
        if relative_chunk == None:
            return None
        contents = self.conmpress_content(relative_chunk,question)
        prompt = template.format(question=question, context=contents, json=js)
        prompt_dict = {}
        prompt_dict["prompt"] = prompt
        result = gpt_request_from_relative_chunk(prompt_dict)
        try:
            json_object = json.loads(result)
            self.data["answer"]["study_type"] = json_object
            return json_object
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None


        

    def gather_details(self):
        return self.data

    def save_to_json(self, filename):
        directory_path = os.path.dirname(filename)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path,exists_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)
