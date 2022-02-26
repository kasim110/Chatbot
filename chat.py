import random
import json
from tkinter import END
from functools import reduce
from itsdangerous import json
from matplotlib.pyplot import title
from numpy import append
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Candi.ai"

dict1 = {"job-title":[],"Skill":[],"Experience":[],"Location":[],"Shift":[]}

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    file = open("msg.txt","a")
    file.writelines(msg)
    file.write('\t')
    file.close()
    
    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                open("msg.txt","w").close()
                return random.choice(intent['responses'])


    

    file1 = open("msg.txt","r")
    for line in file1:
        line = line.lower()
    
    job_title = ['automation test engineer', 'information security engineer', 'business solutions architect', 'java developer (mid level)- ft- great culture, modern technologies, career growth', 'devops engineer', 'sap fico architect', 'network engineer', 'sr. web application developer (cloud team)', 'front end developer', 'application support engineer', 'openstack engineer ', 'data security administrator - unix & iam', 'software engineer manager', 'sales engineer ', 'project manager', 'windows system administrator', 'test lead / test manager', 
        'principal application analyst-supply chain lawson', 'messaging administrator', 'java architect ', 'business analyst', 'sales manager', 'sr. systems test engineer', 'quality consultant', 'usb validation engineer', 'senior product manager', 'frontend/ui developer', 'selenium automation testing', 'fireeeye hx - security engineer', 'sr. software storage engineer', 'c++ software engineer', 'domestic outsourcing business development executive', 'lead mobile product manager', 'project planner/scheduler', 'business systems analyst', 
        'lead java engineer', 'project coordinator', 'core java developer with distributed computing', 'software developer', 'mobile automation tester ', 'project manager : information security and risk management', 'informatica admin', 'software development engineer, big data', 'sr. business data analyst', 'clinical site monitor', 'product qa engineer', 'system support administrator', 'java developer (ecommerce)', 'geolocation engineer', 'director of product development', 'business intelligence development manager', 'senior. net developer (temp-to-perm)', 
        'web developer', 'senior full stack developer', 'jr business analysts', 'manager - gfs', 'java full stack engineer (angular js is must)', 'windows administrator', 'software engineer (algorithm)', 'sr. javascript developer', 'planner/scheduler', 'technical infrastructure project manager', 'business analyst', 'project coordinator', 'sr service delivery systems administrator (devops) ', 'oracle business systems analyst', 'sr. edi business analyst', 'scientific software specialist and ba', 'linux engineer', 'front end-ui developer/ui-web designer', 'it ops support', 
        'senior product manager, pricing - fulfillment by amazon', 'infrastructure production developer', 'c#.net client/server developer', 'software infrastructure c++ developer', 'account manager', 'technical recruiter', 'senior technical writer', 'sr. quality assurance test analyst', 'business systems analyst iii', 'automated test engineer', 'senior drupal developer', 'senior devops engineer (contract)', 'capacity planning engineer - 11350', 'data center virtualization architect', 'procurement system manager', 'sr. information risk management analyst', 'software development engineer', 
        'account coordinator ii', 'technical lead supply chain - 12241', 'c++ software developer for multi-asset risk system', 'senior mysql dba', 'manager of is network engineers', '(us)-program manager senior', 'business analyst - mortgage/equiting lending, lean six sigma green belt/black belt', 'dhmsm operational medicine interface developer', 'information technology architect', 'swift messaging specialist', 'technician-systems', 'angular js / soa / web developer w/ middleware', 'bi developer/architect', 'soarian clinicals consultant (cerner)', 'san storage engineer', 'java developer', 
        'sql web application developer', 'mobile device qa tester ii', 'processor (screen) - 3rd shift', 'sr. application programmer', 'full stack developer', 'technology manager of analytics', 'exhaust processor (exhaust) 3rd shift', 'security - infosec assessment', 'sr. systems engineer - storage (contract-to-hire)', 'sap technical lead (erp, ecc, hana) in melville, ny or davidson, nc', 'audio / visual support tech', 'auditor analyst', 'sr. java developer', 'cobol developer / programmer', 'business analyst - digital analytics, google, adobe, tableau, foresee, sql', 'cyber combat targeteer', 
        'ba/ qa tester', 'business analyst - pci', 'sustaining engineer iii', 'developer - ios, android, rest api, sql, xcode, ajax', 'processor (tube assembly)', 'treasury associate', '.net developer iii ms dynamics', '.net architect / manager', 'director, information security', 'service technicians - printing', 'reporting analyst', 'business development manager', 'it project manager (mobile apps)', 'application architect', 'director of is, infrastructure, network, best practices, it policy and procedures', 'senior java developer - lead', 'lead/sr data scientist', 'computer systems software engineer', 
        'agile developer', 'testing technical specialist', 'manufacturing engineer ii', 'sr. full stack java developer', 'system engineer', 'electronic classroom system administrator', 'tech support engineer', 'java engineer', 'developer - sql, web services, .net', 'embedded software engineer', 'front office developer', 'developer - web - html5, css3, javascript, ajax, jquery, asp.net', 'sharepoint administrator', 'position type: it-contract', 'android developers', 'systems administrator (vmware / red hat linux)', 'program scheduler v', 'sales support analyst', 'information systems security design architect cissp required', 
        'ui/ux designer', 'java application architect', 'business analyst - business analyst healthcare', 'oracle ebusiness programmer (oracle r12)', 'network administrator - cisco ucs, vmware, citrix, windows', 'wireless/rf network engineer', 'data analyst', 'hpc-linux administrator', 'cloud orchestration and automation', 'mechanical engineer', 'developer - .net, build, release, deployment, sql', 'developer - angularjs,angular, bootstrap, node.js, express, javascript,', 'natural/adabas developer', 'facility project manager/coordinator', 'sr. frontend developer (lead)', 'benefits associate i- (junior role)', 'db engineer', 
        'human resources recruiting generalist', 'corporate counsel', 'sr. senior manager of software engineering in austin, tx', 'windows endpoint security solution engineer/architect (finance industry)', 'network monitoring tools specialist', 'web developer (aws and nodejs)', 'counselor- project/program (admin)', 'production shipper', 'peoplesoft elm developer:', 'customer service representative', 'hedge fund developer with c# & asp.net', 'project manager i', 'manager - qa, automation, management, team development, sdlc, agile,soa', 'quality assurance tester', 'configuration build engineer', 'e-commerce application admin', 
        'windows desktop engineer (application integration)', 'application programmer v specialist', 'network engineer/ architect', 'it security analyst', 'quantative analyst', 'systems engineer / systems integrator', 'sr. software engineer / technical lead', 'data architect - iii', 'systems engineer', 'mysql database architect/administrator', 'test lead', 'software engineer-graphics interface development', 'microstrategy developer', 'sr network engineer', 'directory services engineer/ active directory sme', 'senior mainframe test analyst', 'technical business analyst (microsoft exp.)', 'devops', 'linux administrator (scripting must)', 
        'biztalk developer', 'developer - angularjs, spring or struts, restful', 'technical project manager', 'javascript web programmer analyst', 'jda developer', 'developer - php lamp mvc msql ood', 'systems analyst - healthcare claims, erisa, cobra, hipaa. sql, sdlc', 'micro-services developer', 'electrical project engineer', 'sr software staff ui engineer', 'middleware developer', 'senior developer/tech lead', 'it buyer', 'oracle functional consultant', 'business intelligence developer (it)', 'java/j2ee developer (big data / hadoop)', 'java developer, angularjs, nodejs', 'infrastructure project manager', 'senior accounting/audit specialist/msa',
        'business systems analyst ii', 'senior data scientist', 'sr. software engineer iaas hybrid it', 'relationship manager / account manager, mortgage planning expertise', 'human resources generalist - outlook, excel, customer service', 'data scientist', 'building technician iii (ts/sci)', 'recruiting sourcer (cyber)', 'python / django developer', 'assembly technician ii (1st shift)', 'gui developer', 'sr. sql server datawarehouse architect', 'change and release management associate', 'servicenow developer/ pm', 'sr. systems engineer', 'customer service representative senior', 'system security engineer (information assurance, anti-tamper) in nj', 
        'project manager - pm, pmp, sap, project online, ariba', 'software team lead - microsoft technologies', 'lead qa - automation sme - local to san francisco ca only', 'devops staff engineer/system admin- opnfv open lab -contractor', 'dba - oracle, unix, rac ,11gr2,', 'developer - oracle, peoplesoft financials and hr, ms sql server', 'etl developer', 'quality inspector', 'senior python developer', 'senior financial reporting analyst', 'electrical engineer ii', 'cloud infrastructure developer (python)', 'business analyst (iiba certification)', 'pcie design with spyglass', 'entry level business systems analyst', 'labview software programmer', 'sap fico with revenue recognition (12+ years exp)', 
        'enterprise domain architect - supply chain', 'product manager', 'jr level it business analyst', 'full stack software engineer', 'software specialist vi', 'server analyst', 'splunk admin', 'sdet qa automation engineer - local to columbus ohio only', 'vm ware consultant - vcp certified', 'it project manager - reservation systems', 'senior java server side/front end developer', 'test technician', 'technology audit vp', 'project manager - csp, scrum master, certification, coaching,.net', 'solutions architect', 'network / system engineer', 'support engineer - cloud computing', 'it dev | database | level 2', 'java developer (content management)', 'database engineer', 'qa test automation engineer', 
        'pharmacy tech (pd) - walnut creek campus','lead cloud engineer', 'testing sr. analyst', 'big data solution architect - boston, ma or minnetonka, mn', 'okta idam consultant in chicago, il', 'access & identity management (aim) analyst', 'software engineer, hadoop', 'oms (sterling oms) architect/consultant', 'respiratory therapist - walnut creek campus (per diem)', 'solution architect - eden prairie, mn', 'big data applications technical product owner - minnetonka, mn', 'sr web developer - test and measurement', 'mq middleware support', 'enterprise systems engineer', 'qa automation engineer', 'field services analyst iii', 'java developer 12676', 'security engineer', 'senior security architect', 
        'senior consultant w/ top secret clearance', 'sap consultant - consumer goods/packaged meat industry experience', 'senior i.t. systems analyst', 'it - project manager iv', 'salesforce technical lead - sales / service / marketing cloud', 'senior sales engineer -pre-sales- top rated boston tech company!', 'web developer - javascript expert!', 'hybrid mobile developer - html5, css3, angularjs', 'marketing analyst', 'pmo support analyst', 'business intelligence analyst', 'sap vc with ipc - manager (customer capability)', 'java/spring/cloud based web services applications developer', 'senior mechanical engineer - electronics packaging', 'information security manager - it security, cissp, policies', 
        'data communication analyst - senior', 'ruby on rails software engineer- web applications, javascript', 'disaster recovery project manager', 'senior network engineer- bgp, ip network design and architecture', 'insurance consulting leader', 'telecommute backend software engineer - (python, big data)', 'elixir engineer - elixir, python, ruby on rails', 'core java developer with jbpm and multithreading', 'junior qa engineer (1-5 years experience)', 'sap lshc senior manager (finance capability)', 'business systems analyst - supply chain', 'java developer with android', 'information systems security engineer-cissp', 'biomedical engineer - amazing company culture!', 'full stack developer - javascript, c#, php, mysql', 
        'undergrad intern - colleague insights and analytics', 'senior platform software engineer', 'oracle hcm cloud (fusion) senior solution engineer - usdc', 'developer in test', 'sr infrastructure project manager boston ma moving fast', 'multiple ruby on rails, ror web software engineers to lead level', 'big data engineer', 'senior software engineer - growing company & amazing culture', '.net developer', 'lead mobile application developer', 'sap successfactors - senior manager', 'lead .net developer', 'sr. devops engineer - linux, aws, puppet', 'hpc apps/tools specialist', 'automation/control systems engineer', 'senior manager, salesforce crm', 'sap successfactors - senior consultant', 'senior netsuite analyst - netsuite, erp, javascript', 
        'sap developer - 4.x, sql, data modeling', 'senior core java(10 yrs plus)', 'strategy consulting - human capital - hr shared services senior manager', 'ruby on rails developer - rails expertise needed please!!', 'technical architect oaam', 'operations engineer - linux, azure, chef/puppet/ansible/salt/ter', 'sap successfactors - consultant', 'senior software architect - java, saas, big data', 'fraud investigation analyst', 'mainframe developer (no subcontractors)', 'senior web developer - php, fullstack', 'systems administrator - data center operations w/windows & linux', 'senior back end developer - python, big data processing', 'solution architect, master data management (mdm) - east', 'mainframe systems engineer with tn3270 emulator integration_3+ months_franklin, ma (onsite only)', 
        'senior mechanical engineer', 'devops engineer - aws, linux, docker', 'principle sw firmware engineer - & web based tools- too cool!', 'javascript developer - react.js', 'c/c++ software engineer- medical devices/consumer electronics', 'dev ops engineer', 'tech ops technician (1st shift) (thur-mon)', 'senior digital analyst', 'information systems/support engineer- direct hire', 'senior javascript developer', 'sr. project manager', 'senior product manager - b2b, saas, strategy, innovation', 'senior mechanical engineer (remote ok)', 'successfactors sr consultant- senior solution engineer - usdc', 'ux designer - web/mobile ux for growing international firm!', 'cognos developer', 'sr. java developer (local to ma)', 'sap cash management', 'kinaxis - scm', 'big data software engineer - php/java/aws', 
        'associate director of commercial it applications', 'sap successfactors - manager', 'compliance systems analyst', 'account executive - it', 'technical support specialist', '.net c# developer', 'ux designer for leading edtech firm in boston', 'web services developer', 'geotechnical engineer', 'senior informatica developer - boston, ma or rocky hill, ct', 'configuration management analyst - senior', 'enterprise information management (eim) principal consultant (remote)', 'project manager - premier & innovative engineering company!', 'lead android developer - java, android sdk, android', 'vista test engineer', 'back-end java developer atg', 'architect', 'adobe ame architect', 'technology development program associate - boston, ma', 'exchange support engineer', 'software engineer - php, oop, mysql', 
        'technical change management senior consultant', 'senior front end engineer - implement react', 'technical project manager - agile, sql', 'full stack java developer', 'mba intern - enterprise market strategy', 'database engineer/architect', 'automation systems engineer', 'portal support analyst', 'lead android developer for an exciting start-up!', 'network analyst - network administration, windows, sql', 'senior information security specialist - m', 'avm service delivery director', 'principal ruby on rails developer - ror, javascript, jquery, sql', 'director of engineering & technical services', 'qlikview architect / technical expert', 'associate product manager', 'multimedia training developer', 'successfactors functional consultant - solution engineer usdc', 'sofware engineer - nlp, text mining, healthcare informatics', 
        'telecom business analyst (system engineer)', 'help desk / technology support analyst', 'java developer with jenkins/junit', 'mid-level windows devops engineer', 'java lead / java developer with spring', 'sap retail with mm, afs & fms -consultant (customer capability )', '.net developer (junior / mid-level)', 'workday functional consultant - solution engineer - usdc', 'senior systems engineer - powershell, virtualization, microsoft', 'jr./sr./lead java developer', 'senior software engineer - r / rstudio - boston, ma or rocky hill, ct', 'senior developer - heavilyfunded, revolutionary biotech co! relo', 'senior economist - fortune 500 company- media facing- sql/r', 'adobe analytics / ensighten developer', 'ios software engineer', 'java developer boston', 'ak-javaawshadoop', 'ibm rational tool', 'aws cloud engineer', 'platform engineer - aws, devops, php', 
        'senior oracle hcm business analyst', 'data engineer - php/java/aws/emr', 'senior python/django engineer- remote 3 days a week', 'sap technical architect - senior consultant (technology capability)', 'sr. digital project specialist', 'workforce management - solution engineer - usdc', 'principal video processing architect - hot technology!', 'senior etl developer', 'it - project manager iii', 'buyer / planner', 'software engineer - full stack cloud product development', 'sap mm plus pp or wm - consultant (supply chain capability)', 'front end architect - javascript, ember.js, angular.js', 'informatica developer', 'business systems analyst consultant - hartford, ct', 'principal ruby on rails engineer - ruby on rails, javascript,', 'sr. java engineer', 'sap pp, mm or sd with mdg - senior manager (technology capability)', 
        'lead front end engineer - javascript frameworks, mobile/web, res', 'back end software developer - java, sql, linux', 'macintosh/mac support engineer - dedham, ma', 'program manager m&a', 'remote sr. software architect- c++, ibm i', 'ruby on rails developer - groundbreaking application!!', 'application support specialist - travel', 'senior .net software engineer (c#/asp.net mvc, cloud, cms)', 'oracle erp cloud technical (financials/scm)', 'marketing/communications coordinator', 'elixir developer - elixir, ruby on rails, javascript', 'sas analyst', 'backfill opportunity || ivr automation tester|| fulltime', 'psft supplier contracts lead', 'front end web developer (up to $100,000.00 + bonus)', 'javascript developer (angular js)', 'mysql dba', 'senior full stack engineer - rapidly growing creative events co.', 'lead ux designer - ux design, javascript/html/css, node.js', 
        'sap pp, mm or sd with mdg manager (technology capability)', 'crm business analyst', 'bsp lead engineer - relocation offered', 'strategy consulting - human capital - hr transformation manager', 'principal software engineer - javascript, python, ruby on rails', 'solution architect', 'cloud & infrastructure engineer', 'sap plm - senior consultant (supply chain capability)', 'business analyst - speech analytics', 'business information specialist', 'systems administrator - windows, powershell, azure', 'senior ui developer (100% remote)', 'database consultant', 'business systems analyst/ data quality', 'java developer (no subcontractors)', 'money coach (icf)', 'senior software engineer (c++, c# and asp.net, netsuite )', 'magento software developer: relocate to manchester, nh', 'enterprise information management specialist', 'business analyst with data and process/workflow skills', 'node .js consultant', 'sap business objects developer - 1204', 'salesforce business analyst', 'lead network engineer', 'inside sales / business development representative', 'associate network engineer - cisco voip', '.net wpf developer', 'physicist superconducting electronics & advanced computing technolo', 'data engineer - data engineer', 'tech lead - clinical trials', 'java developer (java with aws or java with hadoop)', 
        'strategy consulting - human capital - hr shared services consultant', 'healthcare informatica developer @ boston (locals only -f2f reqd.))', 'technology lead us sharepoint at norwood, ma', 'data software engineer', 'fulltime: network engineer - quincy ma', 'big data architect', 'rack & stack lab administrator', 'android developer', 'sr. manager in application development(director level)', 'ecsi-21202-payroll executive-full time-westborough-ma-ct', 'salesforce architect / developer', 'ruby on rails developer', 'systems architect', 'polymer js developer', 'field operations technician 2', 'senior .net developer', 'product development business analyst', 'gmc inspire consultant', 'manufacturing engineer', 'front end web developer', 'mumps/vista programmer analyst', 'team center admin/developer - remote position', 'data power admin', 'tech lead/software engineers (full stack)', 'senior sdet engineer', 'senior javascript engineer', 'lead ux/ui designer', 'lead bi developer', 'hardware engineer', 'systems administrator / ironport', 'associate citrix engineer', 'software developeer', 'devops lead', 'technical service specialist ii', 'network security administrator', 'sr windows systems administrator (some linux, cisco, vmware, san...)', 'salesforce business analyst - providence,ri for 6+ months', 
        'software engineer/architect', 'sr. full-stack .net developer', 'sr. sw engineer: c#, .net, web services', 'workday integrations - solution engineer - usdc', 'sr. systems analyst', 'agile coach contract', 'iam specialist', 'senior business analyst', 'requirement management engineer at stow,ma', 'salesforce.com developer', 'dot net developers', 'audio/visual technician', 'salesforce solution architect', 'webmethods, webservices, xml, c# developer', 'systems administrator - endpoint', 'business analyst (testing background)', 'sr. electrical engineer - pharmaceutical design', 'staffing account executive', 'swift developer', 'bigdata architect', 'informatica lead', 'front-end developer (sd)', 'senior mechanical designer', 'oracle hcm cloud (fusion) solution developer - usdc', 'software engineer c,c++, doors, clearance, marlborough ma', 'design verification consultant - 000612', 'enterprise project manager', 'asp.net developer', 'successfactors functional analyst - solution developer- usdc', 'ios/point of sale', 'sr.business systems analyst', 'it contract manager', 'software engineer in test (sdet)', 'technical writer', 'strategy consulting - human capital - hr shared services consulting manager', 'splunk administrator / splunk architect', 'back end java developer', 'systems administrator', 'associate application developer', 
        'systems engineer ii emc itil', 'data security analyst', 'eclinicalworks system administrator', 'verification engineer - specialized -000615', 'solution architect v - database platform architect', 'client services account coordinator', 'senior software engineer', 'oracle database administrator', 'sr. business analyst (bi background) for capital markets', 'ui designer', 'java team/technical lead', 'oracle plm project manager', 'application support / data analyst', 'sr. software engineer', 'analytics consulting director - insurance', 'network support / server support / technical support', 'disaster recovery consultant / technical writer', 'email archiving technician', 'software engineers - new rapid prototype team, total greenfields', 'principal software engineer(s) sig. processing and/or dsp (c++, linux, matlab)', 'senior consultant sap security with hana or java stack', 'fulltime position : looking for front end developer', 'director engineering and technical services- broadcast', 'sap bw backend architect', 'principal sw engineer image processing', 'financial analyst', 'jira developers', 'software engineer', 'sr. agile coach', 'senior lead software engineer', 'senior network engineer (network technician 3 or 4), ucp 9 or 11', 'senior javascript ui engineer', 'genesys engineer', 'systems lead', 'lead data center facilities', 
        'software test and evaluation engineer', 'mainframe z/os admin', 'sap vertex remote', 'director of quality assurance', 'software eng - .net, full-stack dev, perm hire with us!', 'senior web application developer - senior web application developer', 'full time bpm aris lead', 'sr. bigdata developer with java exp - lowell ma', 'junior software developer', 'data warehousing developer', 'infrastructure project manager, it', 'storage & linux engineer', 'senior business process analyst', 'sr project manager with financial services and broker dealer experience long term', 'voice engineer (acme / cisco)', 'sr java developer with angularjs, nodejs(right to hire/client cannot transfer/sponsor visa)', 'sr. physical design engineer', 'kony - mobile development architect', 'cloud architect', 'salesforce developer', 'database / etl engineer', 'cloud architect - aws', 'quality assurance engineer', 'citrix xendesktop / packaging engineer (remote option available)', 'business analyst / ux', 'developer, charles river investment management systems', 'associate professional programmer analyst', 'it project manager', 'corporate action system implementation expert', 'sr. system administrator', 'senior programmer analyst', 'mobile ios developer', 'principal blockchain developer', 'principal devops engineer', 'technical liaison', 'sap sd - salesforce integration consultant', 
        'it compliance specialist', 'gis technical consultant needed in waltham, ma', 'full stack .net developer', 'compliance specialist', 'cyber security consultant', 'contract to hire sr microstrategy developer/architect', 'machine learning data scientist - qpid - boston, ma', 'it consultant level iv', 'infrastructure support engineer-cloud', 'azure architect', 'senior jira developer', '.net developer (right to hire/client cannot transfer/sponsor visa)', 'software dev manager, mobile', 'soc security analyst - secureworks providence, ri, atlanta, ga and lisle, il', 'gui tester/automation engineer', 'pl/sql developer', 'sr./mid c# full-stack software engineer (enterprise web apps)', 'sap project manager (who has managed ecc/ crm upgrade)', 'it generalist', 'java senior software engineer', 'senior it project manager: financial services', 'network administrator i', 'salesforce senior manager', 'data architect with retail background', 'user experience designer', 'ux/ui business analyst', 'sales engineer', 'appian admin', 'cache / mumps programmer', 'telecom analyst', 'java developer (full time)', 'ibm z/vm, zlinux mainframe systems programmer (nh)', 'wmb developer', 'ip communications analyst ii', 'qa architect', 'telecom engineer (das design)', 'qa delivery manager', 'assistant director, data analysis', 'data analyst with sql experience', 
        'senior software development project manager', 'cobol programmer with sas', 'workday hcm data conversion- solution engineer - usdc', 'senior information security openings', 'documentum developer', 'lead developer-security team', 'senior oracle ebs administrator', 'copier technical support specialist 2nd level direct hire or contract', 'senior microsoft.net developer wpf', 'cloud infrastructure engineer', 'apm (dynatrace) consultant in hartford, ct', 'php engineer, global security products', 'data migration lead', 'network administrator', 'senior salesforce developer', 'php magento developer', 'program manager (p&c) - full-time / c2h', 'data architect', 'system engineer/vmware', 'sap apo w/ pp ds', 'front end software engineer', 'oracle dba', 'gmc inspire', 'oracle sales cloud functional consultant', 'java architect with aws cloud or big data', 'ios developer', 'technical support manager', 'production support lead', 'sr../princ c++/ui. software engineer', 'vice president of technology digital agency experience', 'linux systems engineer / administrator', 'systems administrator - engineer', 'linux database administrator-full time (remote working opportunity)', 'sap successfactors - solution architect', 'soc analyst', 'systems engineer: networking (contract-to-perm)', 'site reliability engineer', 'gmc inspire developer', 'informatica developer - contract to hire', 'performance engineer', 'financial application integration specialist', 'junior quality assurance (qa) engineer', 'senior scientific programmer/analyst earth & environment boston', 'application security lead / penetration tester', 'sap senior fi/co expert', 'cognos bi engineer (etl)', 'cyber security analyst iii', 'teamcenter administrator/developer', 'software automation lead', 'data management lead', 'professional system programmer analyst', 
        'jr/mid level software engineer', 'hadoop developer (f2f interview required)', 'implementation project manager', 'odi systems analyst/programmer', '"bpm business analyst or project manager"', 'lms (saba) administrator - waltham, ma', 'sr. salesforce technical architect', 'aws architect', 'oracle / ibm / red hat ldap consultant', 'senior scrum master', 'salesforce / sfdc technical architect', 'sr. buyer / technical / new products', 'junior software engineer c# and asp.net', 'cybersecurity analyst, mid', 'windows cloud system administrator', 'ux designer / ui developer', 'desktop support specialist', 'embedded linux engineer', 'embedded software test engineer', 'sr .net developer', 'sap lead analyst', 'exadata dba', 'mainframes z/os admin', 'ui web application architect', 'production planner', 'it business partner, procurement', 'robotic process automation / rpa consultant', 'lead developer/platform architect', 'helpdesk ii', 'multiple web development openings', 'verification engineer', 'senior database administrator', 'perl architect', 'big data consultant', 'business/data analyst', 'build and release engineer(h1 copy must)', 'senior software engineer (java/javascript)', 'java architect', 'desktop support engineer', 'edi analyst', 'system engineer with vmware and emc', 'sr. qa engineer', 'pplus programmer analyst', 'perm senior project manager', 'application development manager direct hire', 
        'database management/administration tier developer', 'software engineer (back-end)', 'php/ magento developer - php/ magento developer', 'architect with application programming on linux platform', 'travel project - iplanet consultant', 'design quality engineer', 'senior software business analyst', 'it help desk representative', 'mechanical design engineer', 'applications developer', 'help desk specialist (secret clearance & security+ ce)', 'workday integrations - senior solution engineer - lake mary - usdc', 'it system analyst c2c or w2', 'senior oracle dba/ sql developer', 'it support specialist', 'senior java engineer', 'automation qa consultant', '.net/c# developer', 'solutions integration consultant', 'application security engineer', 'cobol cics/db2 programmer', 'agile coach project manager', 'software account executve', 'web developers - brand new data driven development', 'erp/crm consultant', '.net developers/leads in hartford, ct', 'sharepoint developer', 'lead ui engineer', 'vantage programmer analyst', '.net software engineer', 'applications developer | level iii', 'oracle ebs portfolio application lead', 'cloud operations manager', 'lead data engineer', 'quality assurance analyst', 'junior systems administrator', 'sap functional consultants', 'sr project manager with financial services - f2f interview required', 'senior global sales eecutive', 'ui developer', 'senior sql developer', 'engineering manager', 'software build and release engineer', 'director application architecture', 'sharepoint 2010 .net developer', 'solution developer', 'sr. technical project manager', 'digital design director', 'software developer, linux kernel', 'cloud services compliance analyst', 'software programmer', 'test architect', 'biologics process operator', 'electrical engineer i/ii', 'database administrator', 
        'application specialist', 'principal software engineer master level expertise in c# and asp.ne', 'senior information security architect (iam & security architecture)', 'project manager/sr. consultant', 'data scientist in waltham ma', 'application support', 'angular js lead at woonsocket, ri / monroeville, pa / northbrook, il', 'principal cloud architect', 'senior security/network engineer', 'mid-level .net web developer', 'data modeler - big data', 'perm healthcare programmer analyst', 'ux researcher', 'tableau developer - 02651', 'security consultant', 'software assurance analyst', 'peoplesoft hcm consultant local to ma', 'principal software engineer', 'senior telecom engineer - cisco uc', 'convergence technician', 'document control specialist (medical devices)', 'process engineer', 'noc manager', 'architect, clinical decision support', 'etl informatica consultant', 'sr. cyber information security administrator (remote)', 'desktop engineer 3', 'excellent opening for asp.net developer', 'sr. business analyst - amisys', 'hyperion essbase and planning', 'build & release/devops engineer', 'agile program level coach', 'ods data architect', 'hl7 consultant', 'integration developer workday', 'hadoop developer', 'qa engineer', 'etl tester', 'senior product manager, cloud backup and disaster recovery', 'senior embedded sw engineer - qnx', 'salesforce developer with steelbrick implementation', 'sr bsa sap(ariba)', 'hr business analyst', 'oracle/perl developer/production analyst', 'lms analyst/administrator', 'sr. mechanical eng - pharmaceutical design', 'scrum master', 'oracle hcm functional time and labor', 'system analyst', 'business analyst with salesforce & veeva crm exp.', 'linux administrator', 'application developer: sql, php, mysql, html, php', 'help desk analyst i', 'ux designer/ interaction designer', 
        'senior it services sales executive - media & entertainment / telecom industry', 'help desk analyst iii', 'administrative support', 'technology analyst', 'software architect for elite hedge fund', 'principal engineer']

    
    skill = ["python","java","php","azure","sql","qa","javascript","cloud","c++",".net","c#","javascript",
    "matlab","angular.js","angular","react","node.js","spring","hibernate","django","flask","html","css",'rest', 'geo fencing', 'core java', 'android sdk', 'web services', 'soap', 'capital markets', 'lending', 'trading solutions', 'financial services', 'investment banking', 'front office trading', 'advertising technology', 'good', 'functional consultant', 'oracle', 'oracle apps', 'quotation', 'email', 'customer', 'complaints', 'follow', 'support', 'order', 'processing', 'prime', 'operator', 'machine', 'haas', 'fanuc', 'cutting', 'tools', 'facilities', 'team', 'leading', 'archibus', 'hiring', 
    'tririga','financial', 'analysis', 'policy', 'development', 'management', 'skills', 'finance', 'accounting', 'project', 'shop', 'quality', 'control', 'ppap', 'civil', 'works', 'engineer', 'engineering', 'system', 'verilog', 'vhdl', 'verification', 'design', 'test', 'planning', 'magellan', 'solution', 'delivery', 'project_delivery', 'project_management', 'project_planning', 'cyber', 'threat', 'technology', 'strategy', 'cobit', 'operations', 'cissp', 'plsql', 'fruad', 'analyst', 'compliance', 'process', 'associate', 'executive', 'credit', 'freshers', 'adobe', 'experience', 'manager', 'adobe_cq', 
    'apache_jackrabbit', 'daily', 'technical', 'incident', 'staff', 'networking', 'telecom', 'optimization', 'blogs', 'architecture', 'mapkit', 'cocoa_touch', 'xcode', 'objective c', 'sqlite', 'pair programming', 'interpersonal skills', 'emailing', 'result oriented', 'database building', 'administrator linux', 'functional testing', 'sql server', 'rdbms', 'netapp', 'linux', 'ultipro', 'ibm build forge', 'ibm rational clearcase', 'release engineering', 'vehicle insurance', 'radifrequency integrated circuit', 'integrated circuit', 'cellular receivers', 'moulds', 'injection moulding', 'plastic', 'plastic moulding', 
    'injection', 'joomla', 'ajax', 'jquery', 'wordpress', 'magento', 'cake php', 'mysql', 'pythonscripting', 'open stack', 'link builder', '.net', 'abap', 'web dynpro', 'fico', 'excel', 'json', 'data modeling', 'sdlc', 'ssis', 'tuning', 'preparation', 'iterative', 'acceptance testing', 'writing', 'software testing', 'java programming', 'excellent communication', 'data structures', 'customer satisfaction', 'telecaller telesales', 'telemarketing service', 'executive voice', 'build', 'release', 'rational', 'concert', 'jenkins', 'clearcase', 'electronics', 'hardware', 'j2ee', 'j2se', 'j2me', 'javase', 'javame', 'knowledge', 
    'asp.net', 'linq', 'windows', 'services', 'passport', 'information', 'outsourcing', 'offshore', 'party', 'reconciliationreceipt', 'commerce', 'recovery', 'accounts', 'nosql', 'hadoop', 'mongodb', 'scala', 'akka', 'kafka', 'cassandra', 'case', 'generalist', 'activities', 'hrbp', 'business', 'partner', 'willingness', 'learn', 'positive', 'attitude', 'flexibility', 'application', 'receptionist', 'front', 'office', 'calls', 'emails', 'sales', 'coordination', 'security', 'risk', 'audit', 'itgc', 'cisa', 'assurance', 'auditing', 'marketing', 'systems', 'selling', 'mass', 'mailing', 'direct', 'life', 'insurance', 'speak', 
    'implementation', 'language', 'javascript', 'html', 'products', 'angularjs', 'ruby', 'rails', 'icc2', 'innovus', 'p&amp;r', 'timing', 'closure', 'encounter', 'first', 'staffing', 'recruitment', 'inbound', 'outbound', 'interviewing', 'configuration', 'google', 'deployment', 'migration', 'specifications', 'database9i', 'software', 'database', 'technologies', 'resource', 'modelling', 'contact', 'personalisation', 'infrastructure', 'taxation', 'taxinn', 'condition', 'types', 'pricing', 'routines', 'standard', 'cycle', 'interface', 'angular.js', 'html5', 'css3', 'zbrush', 'texturing', 'photoshop', 'maya', 'myntra', 'seller', 
    'inside', 'region', 'growth', 'account', 'hacmp', 'lpar', 'assistant', 'asst', 'leader', 'senior', 'parallel', 'computing', 'posix_threads', 'workload_analysis', 'code_optimization', 'cuda', 'budgeting', 'reporting', 'nagios', 'apache', 'ubuntu', 'fedora', 'fitter', 'institute', 'electrical', 'regional building', 'activation', 'reverse', 'logistics', 'count', 'reefer', 'operation', 'inventory', 'tranportation', 'secretarycs', 'power-(transmission', 'distribution', 'substation)', 'workers', 'compensation', 'research', 'general', 'ledger', 'receivable', 'international', 'gaap', 'transfer', 'standards', 'chartered', 'accountant', 
    'forex', 'treasury', 'tech', 'collection', 'hardcore', 'visa', 'manager(customer', 'relation)', 'coordinator', 'client', 'acquisition algorithm', 'object oriented', 'solutions', 'tele', 'call center', 'telemarketing', 'java', 'hplc', 'finished', 'stability', 'equity', 'handling', 'heading', 'branch', 'relationship', 'stock', 'broking', 'spring', 'hibernate', 'maven', 'ccna', 'troubleshooting', 'mcse', 'coding', 'mathematics', 'science.', 'primary', 'teacher', 'maths', 'teaching', 'graduatemetro operationsquality assurance', 'singur', 'plant', 'turbine manufacturing', 'production', 'vmware', 'performance', 'desk', 'guest', 'relation receptionist', 
    'grass', 'recruiting', 'restful', 'applications', 'native', 'algorithms', 'android', 'structures', 'medical', 'jobs', 'jobsmedical', 'coder', 'nursing', 'pharma', 'biotechnology', 'corporate', 'social media publicity', 'tablets', 'ndds', 'capsule', 'pellets', 'solid', 'oral', 'dosage', 'opentext', 'invoice_management', 'invoices', 'sap_workflow', 'sap_best_practices', 'purchase_orders', 'sap_alv', 'governance', 'module', 'apps', 'msbuild', 'classic', 'ultrasound', 'typist', 'bloomberg', 'terminal', 'solaris', 'underwriting', 'metro', 'content', 'review', 'science', 'applicationxtender', 'junit', 'core', 'forensics', 'encase', 'gcfa', 'electronic_discovery', 
    'investigation', 'incident_management', 'seim', 'android developers', 'python', 'php', 'azure', 'sql', 'qa', 'cloud', 'c++', 'c#', 'matlab', 'angular', 'react', 'node.js', 'django', 'flask', 'css']

    location = ['atlanta','new york','las vegas', 'chicago', 'schaumburg', 'bolingbrook', 'seattle', 'sunnyvale', 'highlands', 'portland', 'hillsboro', 'beaverton', 'kansas', 
    'denver', 'sandy', 'parsippany', 'eden', 'austin', 'columbia', 'philadelphia', 'mountain', 'redmond', 'tampa', 'eagan', 'miami', 'tucson',
    'arlington', 'lisle', 'saint', 'bridgewater', 'chandler', 'oceanside', 'gardena', 'spokane', 'westlake', 'burbank', 'brentwood', 'addison', 
    'camden', 'plano', 'salem', 'bathesa', 'foster', 'pleasanton', 'westmont', 'albany', 'woodland', 'vienna', 'alpharetta', 'richardson', 'ontario',
    'montreal', 'emeryville', 'bellevue', 'tempe', 'jersey', 'vernon', 'melville', 'herndon', 'glendale', 'anaheim', 'edgewood', 'clearwater', 'cypress',
    'havertown', 'folsom', 'irvine', 'sherman', 'washington', 'springfield', 'irving', 'clearfield', 'poway', 'pensacola', 'dulles', 'nashville', 'bethesda',
    'bloomfield', 'simi', 'palm', 'phoenix', 'portage', 'pittsburgh', 'woburn', 'sterling', 'jacksonville', 'baltimore', 'stamford', 'clarks', 'scottsdale', 
    'milwaukee', 'houston', 'laurel', 'sacramento', 'manhattan', 'getzville', 'frederick', 'milpitas', 'raleigh', 'newport', 'reston', 'virginia', 'annapolis', 'needham', 
    'mason', 'oakdale', 'clifton', 'charlotte', 'orlando', 'broken', 'huntsville', 'lake', 'columbus', 'cincinnati', 'paramus', 'ewing', 'lenexa', 'wilmington', 'walnut', 'boston', 
    'minnetonka', 'minneapolis', 'research', 'ashburn', 'monterey', 'colesville', 'cambridge', 'waltham', 'framingham', 'quincy', 'marlborough', 'cumberland', 'hartford', 'andover', 'hanscom',
    'hopkinton', 'woonsocket', 'southborough', 'ellington', 'westford', 'watertown', 'somerville', 'franklin', 'chelmsford', 'bedford', 'burlington', 'lexington', 'malden', 'unspecified', 'billerica', 'hampden', 'manchester', 'peabody', 'beverly', 'newburyport', 'middleboro', 'glastonbury', 'windsor', 'warwick', 'dover', 'lowell', 'brighton', 'dedham', 'nashua', 'exeter', 'wellesley', 'natick', 'providence', 'groton', 
    'norwood', 'brockton', 'westborough', 'worcester', 'portsmouth', 'littleton', 'middletown', 'marlboro.', 'durham', 'stow', 'weston', 'norwell', 'acton', 'wakefield', 'boxborough', 'keene', 'norwich', 'holyoke', 'storrs', 'dorchester', 'medford', 'hudson', 'webster', 'ayer', 'concord', 'norton', 'sudbury', 'milford', 'westbrook', 'oxford', 'mansfield', 'maynard', 'devens', 'westminster', 'smithfield', 'lawrence', 
    'charlestown', 'raynham', 'danvers', 'westboro', 'dresher', 'taunton', 'clinton', 'mashantucket', 'agawam', 'cranston', 'weymouth', 'meredith', 'haverhill', 'simsbury', 'southboro', 'methuen', 'auburn', 'northborough', 'chelsea', 'brattleboro', 'woods', 'nantucket', 'ipswich', 'lincoln', 'holliston', 'claremont', 'newington', 'plymouth', 'dallas', 'coppell', 'frisco', 'denison', 'allen', 'southlake', 'scarborough', 'lewisville', 'fort', 'carrollton', 'hurst', 'tewksbury', 'farmers', 'corinth', 'flower', 'westfield', 'hooksett', 'braintree', 'university', 'merrimack', 'willow', 'grand', 'roanoke', 'contoocook', 'erlanger', 'grapevine', 'fall', 'newark', 'statesville', 'greenville', 
    'rancho', 'campbell', 'greensboro', 'mooresville', 'winston', 'iselin', 'basking', 'mahwah', 'edison', 'princeton', 'roseland', 'bristol', 'hopewell', 'lansdale', 'trumbull', 'norwalk', 'elmwood', 'ridgewood', 'rockaway', 'piscataway', 'lebanon', 'wayne', 'bridgeport', 'plainsboro', 'secaucus', 'hamilton', 'centerbrook', 'pennington', 'warren', 'plainview', 'cranford', 'mount', 'freehold', 'hauppauge', 'pompton', 'malvern', 'jenkintown', 'great', 'allentown', 'collegeville', 'voorhees', 'cherry', 'brooklyn', 'mineola', 'holmdel', 'florham', 'ridgefield', 'southampton', 'marlton', 'oakland', 'garden', 'trevose', 'bernardsville', 'berkeley', 'farmington', 'hoboken', 'newtown', 'livingston', 'wilton', 'ramsey',
    'astoria', 'yardley', 'westchester', 'morristown', 'somerset', 'bethlehem', 'danbury', 'whippany', 'weehawken', 'raritan', 'bronx', 'jamaica', 'rahway', 'englewood', 'millburn', 'pearl', 'trenton', 'branchburg', 'radnor', 'woodbridge', 'orange', 'langhorne', 'orangeburg', 'port', 'paulsboro', 'stony', 'cranbury', 'rochelle', 'morris', 'horsham', 'tarrytown', 'oaks', 'chester', 'hicksville',
    'wallingford', 'union', 'glen', 'monmouth', 'devon', 'rockleigh', 'conshohocken', 'ambler', 'titusville', 'manasquan', 'montvale', 'summit','allendale', 'cedar', 'atlantic', 'commack', 'meriden', 'hackensack', 'exton', 'southbury', 'albertson', 'murray', 'purchase', 'greenwich', 'bedminster', 'bala', 'teterboro', 'branford', 'westbury', 'queens', 'doylestown', 'wawa', 'bethpage', 'norristown', 'shelton', 'villanova', 'berlin', 'yonkers', 'woodcliff', 'floral', 'fairfield', 'moorestown', 'coopersburg', 'riverdale', 'bethel', 'wethersfield', 'roseville', 'fremont', 'menlo', 'mill', 'cupertino', 'modesto', 'novato', 'bayport', 'warrington', 'harrison', 'jericho', 'manayunk', 'seaside', 'valhalla', 'teaneck', 
    'benicia', 'holbrook', 'mcclellan', 'redwood', 'flemington', 'mather', 'palo', 'lyndhurst', 'corte', 'rutherford', 'kenilworth', 'essington', 'jamesburg', 'alameda', 'stanford', 'burlingame', 'hackettstown', 'royersford', 'little', 'richmond', 'kennett', 'culver', 'bakersfield', 'naperville', 'madison', 'owings','kirkland', 'tumwater', 'bothell', 'tacoma', 'edmonds', 'renton', 'lacey', 'issaquah', 'kent', 'everett', 'seatac', 'puyallup', 'federal', 'mountlake','spring', 'pearland', 'bunker', 'stafford', 'bainbridge', 'ferndale', 'lynnwood', 'buffalo', 'piney', 'bryan', 'snoqualmie', 'beaumont', 'lakewood', 'magnolia', 'nassau', 'tukwila', 'bremerton', 'torrance', 'costa', 'wadsworth', 'katy', 'bellaire', 'carlsbad', 
    'huntington', 'riverside', 'pasadena', 'galveston', 'newbury', 'buena', 'temecula', 'humble', 'cerritos', 'dickinson', 'edwards', 'thousand', 'united', 'memphis', 'alexandria', 'indianapolis', 'moline', 'painted', 'dublin', 'melbourne', 'detroit', 'rockville', 'charleston', 'rochester', 'monroe', 'mexico', 'iowa', 'deerfield', 'sidney', 'chattanooga', 'cary', 'lenoir', 'aberdeen', 'georgetown', 'flanders', 'lawrenceville', 'norfolk', 'greenbelt', 'peachtree', 'johns', 'smyrna', 'gainesville', 'decatur', 'athens', 'norcross', 'duluth', 'buckhead', 'pendergrass', 'oakwood', 'marietta', 'kennesaw', 'covington', 'warner', 'college', 'cumming', 'dunwoody', 'suwanee', 'ball', 'macon', 'eatonton', 'buford', 'bogart', 'doraville', 
    'roswell', 'forsyth', 'white', 'lithonia', 'austell', 'mclean', 'crownsville', 'falls', 'fairfax', 'york', 'towson', 'gaithersburg', 'mechanicsville', 'chantilly', 'ellicott', 'middle', 'lanham', 'parkville', 'hanover', 'hershey', 'merrifield', 'hollywood', 'linthicum', 'silver', 'germantown', 'camp', 'tysons', 'harrisburg', 'lorton', 'suitland', 'chevy', 'sarasota', 'adelphi', 'belcamp', 'charles', 'crystal', 'bolling', 'acworth', 'pikesville', 'quicksburg', 'manassas', 'cheaspeak', 'sparks', 
    'dearborn', 'carmel', 'cleveland', 'omaha', 'lafayette', 'waukesha', 'islandia', 'wauwatosa', 'skokie', 'mayfield', 'bloomington', 'neenah', 'coral', 'rockford', 'louisville', 'holtsville', 'largo', 'colorado', 'corning', 'lansing', 'plantation', 'glenview', 'sioux', 'greenwood', 'boise', 'overland', 'broomfield', 'odenton', 'fernley', 'saline', 'richfield', 'syracuse', 'birmingham', 'centennial', 'martinez', 'augusta', 'cheverly', 'toronto', 'carson', 'livonia', 'waterbury',  'topeka', 'twin', 'naples', 'westwood', 'tulsa', 'niles', 'roseburg', 'oakbrook', 'marina', 'southfield','ottawa', 'hyderabad', 'nevada', 'honolulu', 'cheswick', 'boca', 'fresno', 'dahlgren', 'mounds', 'dania', 'bangalore', 'tuscaloosa', 'mississauga', 'edmonton', 
    'cuyahoga', 'midland', 'thomasville', 'albuquerque', 'verona', 'marshalltown', 'shawnee', 'point', 'bolivar', 'strongsville', 'morrisville', 'thanksgiving', 'clayton', 'prairie', 'chesterfield', 'peterborough', 'stockton', 'sylmar', 'south', 'ogden', 'patuxent', 'oklahoma', 'ofallon', 'royal', 'district', 'halifax', 'quantico', 'encinitas', 'venice', 'pleasant', 'draper',
    'moon', 'center', 'hoffman', 'derwood', 'florence', 'beford', 'council', 'dexter', 'reno', 'tallahassee', 'falmouth', 'georgia', 'maple', 'gurnee', 'phila', 'tinton', 'akron', 'briarwood', 'aliso', 'lakeside', 'muscatine', 'mattawan', 'ronkonkoma', 'minerva', 'roxboro', 'hingham', 'thurmont', 'scott', 'tolleson', 'boone', 'syosset', 'valley', 'woodbury', 'sharon', 'edgewater', 'metuchen', 'picatinny', 'colmar', 'elmsford', 'harleysville', 'broomall', 'woodcrest', 'eatontown','bryn','poughkeepsie', 'cromwell', 'huntingdon', 'hasbrouck', 'westport', 'audubon', 'maplewood', 'totowa', 'peapack', 'briarcliff', 'telford', 'skillman', 'newburgh', 'pottstown', 'wappingers', 'chesterbrook', 'avon', 'nyack', 'ardmore', 'montebello', 'macungie', 'cheshire', 'irvington', 'rocklin', 'delhi', 'mendham', 'delaware', 'hercules', 'hayward', 
    'bensalem', 'sausalito', 'stratford', 'watsonville', 'davis', 'rohnert', 'unionville', 'carlstadt', 'livermore', 'danville', 'johnston', 'century', 'redlands', 'redondo', 'valencia', 'azusa', 'palmdale', 'marysville', 'jefferson', 'blythewood', 'jeffersonville', 'hines', 'scottdale', 'chico', 'lakehurst', 'londonderry', 'greenfield', 'vancouver', 'alhambra', 'miamisburg', 'lombard', 'northampton', 'lodi', 'kaysville', 'boulder', 'juno', 'round',
    'bowie', 'northbrook', 'montgomery', 'lockport', 'hagerstown', 'blacksburg', 'knoxville', 'menomonee', 'itasca', 'oakton', 'prineville', 'hilliard', 'cheyenne', 'temple', 'reading', 'matawan', 'warrenton', 'idaho', 'presidio', 'cheektowaga', 'harlingen', 'west', 'mossville', 'bensenville', 'champaign', 'bentonville', 'roslyn', 'milan', 'blue', 'henderson', 'guerneville', 'davenport', 'rosemont', 'lehi', 'beachwood', 'chatsworth', 'amherst', 'scranton', 'rockwall', 'toledo', 'loveland', 'carteret', 'independence', 'sunset', 'beijing', 'evanston', 'signal', 'woodlawn', 'aurora', 'tustin', 'fishers', 'dayton', 'belton', 'novi', 'lincolnshire', 'butler', 'libertyville', 'davie', 'downers', 'ephrata', 'riverwoods', 'monrovia', 'lusby', 'keller', 'dimondale', 'brainerd', 'moosic', 'diamond', 'hazelwood', 
    'golden', 'santa', 'hollister', 'citrus', 'rosemead', 'downey','fullerton', 'shreveport', 'hickory', 'erie', 'olathe', 'brisbane', 'urbandale', 'mechanicsburg', 'mercer', 'encino', 'eugene', 'brampton', 'sheboygan', 'corona', 'orem', 'pittsford', 'calgary', 'lancaster', 'markham', 'wilkes', 'centreville', 'middlebury', 'lemoore', 'pune', 'edina', 'fredericksbrg', 'crane', 'shorewood', 'highland', 'twinsburg', 'peoria', 'sunrise', 'miramar', 'parsippany troy', 'charlottesville', 'quebec', 'longmont', 'medley', 'kohler', 'clarksburg', 'demotte', 'leesburg', 'oconomowoc', 'kenosha', 'elgin', 'burr', 'rolling', 'tucker', 'sunapee', 'gresham', 'baton', 'canonsburg', 'brookfield', 'chaska', 'calvert', 'lititz', 'belmont', 'warrenville', 'mequon', 'schiller', 'janesville', 'waukegan', 'joliet', 'lemont', 'greensburg', 'beltsville', 'caledonia', 'lewiston', 'alcoa', 'arden', 'northlake', 'binghamton', 'doral', 'broadview', 'tyson s', 'seal', 'utica', 'suffield', 'murrysville', 'frankfort', 'centerville', 'metairie', 'sayre', 'hammond', 'commerce', 'beavercreek', 'cape', 'ossining', 'pompano', 'anchorage', 'okemos', 'macomb', 'brandon', 
    'coconut', 'kentfield', 'murfreesboro', 'mckinney', 'wixom', 'hood', 'cockeysville', 'laguna', 'fountain', 'davidson', 'elmhurst', 'goodyear', 
    'norman', 'lynn', 'tracy', 'stoughton', 'mumbai', 'falconer', 'chula', 'playa', 'grafton', 'bend', 'superior', 'castle', 'konstanz', 'chandigarh', 'ames', 'nanuet', 
    'urbana', 'elkridge', 'corvallis', 'allegan', 'gray', 'bowling', 'boyce', 'central', 'evansville', 'cicero', 'mettawa', 'flint', 'cartersville', 'wichita', 'lindon', 'bridgeville', 'marshall', 'chesapeake', 'rome', 'guantanamo', 'calabasas', 'selden', 'etters', 'beech', 'gastonia', 'ridgeland', 'byhalia', 'goodlettsville', 'kapolei', 'northfield', 'palos', 'abbott', 'kankakee', 'midlothian', 'hatboro', 'stanhope', 'mesa', 'richland', 'taylor', 'hunt', 'oxnard', 'middlesex', 'niwot', 'malta', 'hermitage', 'owatonna', 'forest', 'fenton', 'hopkins', 'benton', 'twentynine', 'tarzana', 'wagener', 'williamsville', 'irwindale', 'beaufort', 'cottleville', 'thornton', 'saint', 'fruitland', 'raytown', 'brown', 'westerville', 'pullman', 'seguin', 'bridgeton', 'petaluma', 'chanhassen', 'leawood', 'hesston', 'radford', 'warrendale', 'kennedy', 'holland', 'lorain', 'bannockburn', 'earth', 'middleton', 'oshkosh', 'brea', 'ventura', 'loma', 'beresford', 'gulfport', 'elkhorn', 'hastings', 'jackson', 'kearneysville', 'roselle', 'dauphin', 'merrillville', 'wilkes barre', 'vineland', 'gilroy', 'napa', 'morgan', 'salinas', 'half', 'scotts', 'saratoga', 'gold', 'tiburon', 'merced']


    experience = ['1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10', '10.5', '11', '11.5', '12', '12.5', '13', '13.5', '14', '14.5', '15', '15.5', '16', '16.5', '17', '17.5', '18', '18.5', '19', '19.5', '20', '20.5', '21', '21.5', '22', '22.5', '23', '23.5', '24', '24.5', '25', '25.5', '26', '26.5', '27', '27.5', '28', '28.5', '29', '29.5', '30']
    shift = ["full time","fulltime","parttime","part time"]


    if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1 and len(dict1['job-title']):
        y = "yes"
        n = "no"
        if y in line:
            file2 = open("info.json","w")
            json.dump(dict1,file2,indent=4)
            file2.close()
            dict1["Skill"].clear()
            dict1["Location"].clear()
            dict1["Experience"].clear()
            dict1["Shift"].clear()
            dict1['job-title'].clear()
            open("msg.txt","w").close()
            return intents['done']
        elif n in line:
            skill1 ="skill" 
            location1 ="location"
            experience1 ="experience"
            shift1="shift"
            title = "job title"
            if skill1 in line:
                for j in skill:
                    if j in line:
                        dict1["Skill"].clear()
                        dict1['Skill'].append(j)
                        file2 = open("info.json","w")
                        json.dump(dict1,file2,indent=4)
                        file2.close()
                        dict1["Skill"].clear()
                        dict1["Location"].clear()
                        dict1["Experience"].clear()
                        dict1["Shift"].clear()
                        dict1['job-title'].clear()
                        open("msg.txt","w").close()
                        return random.choice(intents['skill-submit'])
                return random.choice(intents["ask-skill"])
            elif location1 in line:
                for j in location:
                    if j in line:
                        dict1["Location"].clear()
                        dict1['Location'].append(j)
                        file2 = open("info.json","w")
                        json.dump(dict1,file2,indent=4)
                        file2.close()
                        dict1["Skill"].clear()
                        dict1["Location"].clear()
                        dict1["Experience"].clear()
                        dict1["Shift"].clear()
                        dict1['job-title'].clear()
                        open("msg.txt","w").close()
                        return random.choice(intents['location-submit'])
                return random.choice(intents["ask-location"])
            elif experience1 in line:
                for j in experience:
                    if j in line:
                        dict1["Experience"].clear()
                        dict1['Experience'].append(j)
                        file2 = open("info.json","w")
                        json.dump(dict1,file2,indent=4)
                        file2.close()
                        dict1["Skill"].clear()
                        dict1["Location"].clear()
                        dict1["Experience"].clear()
                        dict1["Shift"].clear()
                        dict1['job-title'].clear()
                        open("msg.txt","w").close()
                        return random.choice(intents['exp-submit'])
                return random.choice(intents["ask-exp"])
            elif shift1 in line:
                for j in shift:
                    if j in line:
                        dict1["Shift"].clear()
                        dict1['Shift'].append(j)
                        file2 = open("info.json","w")
                        json.dump(dict1,file2,indent=4)
                        file2.close()
                        dict1["Skill"].clear()
                        dict1["Location"].clear()
                        dict1["Experience"].clear()
                        dict1["Shift"].clear()
                        dict1['job-title'].clear()
                        open("msg.txt","w").close()
                        return random.choice(intents['shift-submit'])
                return random.choice(intents["ask-shift"])
            elif title in line:
                for j in job_title:
                    if j in line:
                        dict1['job-title'].clear()
                        dict1['job-title'].append(j)
                        file2 = open("info.json","w")
                        json.dump(dict1,file2,indent=4)
                        file2.close()
                        dict1["Skill"].clear()
                        dict1["Location"].clear()
                        dict1["Experience"].clear()
                        dict1["Shift"].clear()
                        dict1['job-title'].clear()
                        open("msg.txt","w").close()
                        return random.choice(intents['title-submit'])
                return random.choice(intents['ask-title'])
            return intents['what']
        return intents['start']

    
    for j in skill:
        if j in line:
            if j not in dict1["Skill"]:
                    a = j
                    print(a)
                    dict1['Skill'].append(a)
    for j in location:
        if j in line:
            if j not in dict1["Location"]:
                a = j
                print(a)
                dict1['Location'].append(a)
    for j in experience:
        if j in line:
            if j not in dict1["Experience"]:
                a = j
                print(a)
                dict1['Experience'].append(a)
    for j in shift:
        if j in line:
            if j not in dict1["Shift"]:
                a = j
                print(a)
                dict1['Shift'].append(a)
    for j in job_title:
        if j in line:
            if j not in dict1['job-title']:
                a = j
                print(a)
                dict1['job-title'].append(a)

    
   
    print(dict1)
      #for Skill scope 
    if len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["Experience"])==0 and len(dict1["Shift"])==0 and len(dict1['job-title'])==0:
        return intents['msg']
    elif len(dict1["Skill"])==0 and len(dict1["Experience"])==0 and len(dict1["Location"])==0 and len(dict1['job-title'])==0 :
        return random.choice(intents['skill-loc-exp-ti'])
    elif len(dict1["Experience"])==0 and len(dict1["Location"])==0 and len(dict1["Shift"])==0 and len(dict1['job-title'])==0:
        return random.choice(intents['loc-exp-shi-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Experience"])==0 and len(dict1["Shift"])==0 and len(dict1['job-title'])==0:
        return random.choice(intents['skill-exp-shift-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["Shift"])==0 and len(dict1['job-title'])==0:
        return random.choice(intents['skill-loc-shift-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["Experience"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['skil-exp-loc-shi'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["Experience"])==0:
        return random.choice(intents['skil-exp-loc'])
    elif len(dict1["Skill"])==0 and len(dict1["Shift"])==0 and len(dict1["Experience"])==0:
        return random.choice(intents['skil-exp-shi'])
    elif len(dict1["Skill"])==0 and len(dict1["job-title"])==0 and len(dict1["Experience"])==0:
        return random.choice(intents['skil-exp-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['skil-loc-shi'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['skil-loc-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Shift"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['skil-shi-ti'])
    elif len(dict1["Skill"])==0 and len(dict1["Experience"])==0:
        return random.choice(intents['skill-exp'])
    elif len(dict1["Skill"])==0 and len(dict1["Location"])==0:
        return random.choice(intents['skill-loc'])
    elif len(dict1["Skill"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['skill-shift'])
    elif len(dict1["Skill"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['skill-ti'])
    elif len(dict1["Skill"])==0:
        return random.choice(intents['skill'])


       #for Experience scope
    if len(dict1["Location"])==0 and len(dict1["Experience"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['exp-loc-shi'])
    elif len(dict1["Location"])==0 and len(dict1["Experience"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['exp-loc-ti'])
    elif len(dict1["Location"])==0 and len(dict1["job-title"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['loc-shi-ti'])
    elif len(dict1["Experience"])==0 and len(dict1["job-title"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['exp-shi-ti'])
    elif len(dict1["Experience"])==0 and len(dict1["Location"])==0:
        return random.choice(intents['loc-exp'])
    elif len(dict1["Experience"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['exp-shift'])
    elif len(dict1["Experience"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['exp-ti'])
    elif len(dict1["Experience"])==0:
        return random.choice(intents['experience'])

        #for Location scope
    if len(dict1["Location"])==0 and len(dict1["Shift"])==0 and len(dict1['job-title'])==0:
        return random.choice(intents['loc-shi-ti'])
    elif len(dict1["Location"])==0 and len(dict1["Shift"])==0:
        return random.choice(intents['loc-shift'])
    elif len(dict1["Location"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['loc-ti'])
    elif len(dict1["Location"])==0:
        return random.choice(intents['location'])

        #for shift scope
    if len(dict1["Shift"])==0 and len(dict1["job-title"])==0:
        return random.choice(intents['shi-ti'])
    elif len(dict1["Shift"])==0:
        return random.choice(intents['shift'])
        #for job title
    if len(dict1['job-title'])==0:
        return intents['title']



    


    if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1 and len(dict1['job-title'])>=1:
        open("msg.txt","w").close()
        return intents['start']

    # if prob.item() > 0.75:
    #     for intent in intents['intents']:
    #         if tag == intent["tag"]:
    #             return random.choice(intent['responses'])
    
    
    
    
    
    return "I don't undertand..."






#  #For making changes  and restart the program
    # if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
        
    #     y = 'yes'
    #     n = 'no'
    #     yes="y"
    #     no="n"
    #     skill1 ="skill" 
    #     location1 ="location"
    #     experience1 ="experience"
    #     shift1="shift"
       
    #     if y in line:
    #         if y in line:
    #             if skill1 in line:
    #                 for j in skill:
    #                     if j in line:
    #                         dict1["Skill"].pop()
    #                         dict1['Skill'].append(j)
    #                         file2 = open("test1.json","w")
    #                         json.dump(dict1,file2,indent=4)
    #                         file2.close()
    #                         return intents['skill-submit']
    #                 return intents["ask-skill"]
    #             elif location1 in line:
    #                 for j in location:
    #                     if j in line:
    #                         dict1["Location"].pop()
    #                         dict1['Location'].append(j)
    #                         file2 = open("test1.json","w")
    #                         json.dump(dict1,file2,indent=4)
    #                         file2.close()
    #                         return intents['location-submit']
    #                 return intents["ask-location"]
    #             elif experience1 in line:
    #                 for j in experience:
    #                     if j in line:
    #                         dict1["Experience"].pop()
    #                         dict1['Experience'].append(j)
    #                         file2 = open("test1.json","w")
    #                         json.dump(dict1,file2,indent=4)
    #                         file2.close()
    #                         return intents['exp-submit']
    #                 return intents["ask-exp"]
    #             elif shift1 in line:
    #                 for j in shift:
    #                     if j in line:
    #                         dict1["Shift"].pop()
    #                         dict1['Shift'].append(j)
    #                         file2 = open("test1.json","w")
    #                         json.dump(dict1,file2,indent=4)
    #                         file2.close()
    #                         return intents['shift-submit']
    #                 return intents["ask-shift"]
    #             return intents['what']
                    
        

    #         elif n in line:
    #             return intents['submit']
    #         return intents['changes']
   
    #     elif n in line:
    #         if yes in line:
    #             if yes in line:
    #                 dict1["Skill"].pop()
    #                 dict1["Location"].pop()
    #                 dict1["Experience"].pop()
    #                 dict1["Shift"].pop()
    #                 print(dict1)
    #                 file1.close()
    #                 open("append.txt","w").close()
    #                 return intents["msg"]
    #             elif no in line:
    #                 open("append.txt","w").close()
    #                 exit()
    #             elif len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
    #                 return intents["repeat"]
    #         return intents["restart"]
    #     return intents['changes']



      #For restart the program
    # if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
    #     y = "yes"
    #     n = "no"
    #     if y in line:
    #         dict1["Skill"].pop()
    #         dict1["Location"].pop()
    #         dict1["Experience"].pop()
    #         dict1["Shift"].pop()
    #         print(dict1)
    #         file1.close()
    #         open("append.txt","w").close()
    #         return intents["msg"]
    #     elif n in line:
    #         open("append.txt","w").close()
    #         exit()
    #     elif len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
    #         return intents["repeat"]
    #     return intents["restart"]
    


    # file2 = open("test1.json","w")
    # json.dump(dict1,file2,indent=4)
    # file2.close()
    # print(dict1)

   
    

    # if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
    #     return intents['restart']

    # if len(dict1["Skill"])>=1 and len(dict1["Location"])>=1 and len(dict1["Experience"])>=1 and len(dict1["Shift"])>=1:
    #     open("append.txt","w").close()
    #     return intents['changes']