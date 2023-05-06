from flask import Flask, render_template, jsonify, request, got_request_exception
import config
import openai
from openai.error import RateLimitError,AuthenticationError,Timeout,APIError,APIConnectionError,InvalidRequestError,ServiceUnavailableError
import os
import rollbar
import rollbar.contrib.flask
import sqlite3
import datetime
import traceback
import sys
api_key=config.DevelopmentConfig.OPENAI_KEY
openai.api_key= api_key


#connect to db
def get_db_connection():
    conn = sqlite3.connect('./database/maestro.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor() 
    cur.execute("PRAGMA foreign_keys = ON;")
    return conn

def page_not_found(e):
  return render_template('404.html',page="Chatbot project",**locals()), 404

app = Flask(__name__)
app.config.from_object(config.config['development'])
app.register_error_handler(404, page_not_found)

##send errors to rollbar for analysis
@app.before_first_request
def init_rollbar():
    """init rollbar module"""
    rollbar.init(
        # access token
        '773addfb39d84682890828b418f2073f',
        # environment name
        'development',
        # server root directory, makes tracebacks prettier
        root=os.path.dirname(os.path.realpath(__file__)),
        # flask already sets up logging
        allow_logging_basic_config=False)

    # send exceptions from `app` to rollbar, using flask's signal system.
    got_request_exception.connect(rollbar.contrib.flask.report_exception, app)

@app.route('/')
def index():
    return render_template('index.html',page="ChatGPT Project",**locals())

@app.route('/setup.html', methods=['POST', 'GET'])
def setup():
    if request.method == 'POST':
        content = request.form['content']
        save_system_instruction(content)
    return render_template('setup.html',page="Setup ChatGPT",**locals())

@app.route('/chatbot.html', methods=['POST', 'GET'])
def chatbot():
   if request.method == 'POST':
        user_input = request.form['prompt']
        messages = []
        #sets the role of chatgpt
        #system: facts, basic and important context (no behaviour instructions here): example: At this place apples are purple.
        messages.append({"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-04-10"})
        question = {}
        question['role'] = 'user'
        question['content'] = user_input
        messages.append(question)
        try:
            response = openai.ChatCompletion.create(
                # model="gpt-4",
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5
            )
            # content = response.choices[0].message["content"]
            content = response['choices'][0]['message']['content'].replace('\n', '<br>')
            count_tokens = response['usage']['total_tokens']
        except Exception as e:
            # monitor exception using Rollbar
            rollbar.report_exc_info()
        except RateLimitError:
            content = "You have hit your assigned rate limit. "
        except AuthenticationError:
            content = "Your API key or token was invalid, expired, or revoked."
        except Timeout :
            content = "Request timed out."
        except APIError :
            content = "The server is experiencing a high volume of requests. Please try again later. Issue on our side."
        except APIConnectionError:  
            content = " Issue connecting to our services."
        except InvalidRequestError :
            content = "Your request was malformed or missing some required parameters, such as a token or an input."
        except openai.error.PermissionError :
            content = "Your API key or token does not have the required scope or role to perform the requested action."
        except ServiceUnavailableError :
            content = "Issue on our servers."
        finally:
            save_response_token(count_tokens)
        

        return jsonify(content=content,count_tokens=count_tokens), 200
   
   return render_template('chatbot.html', page="Chatbot project",**locals())

@app.route('/chatbotimage.html', methods=['POST', 'GET'])
def chatbotimage():
   if request.method == 'POST':
        user_input = request.form['prompt']
        messages = []
        #sets the role of chatgpt
        #system: facts, basic and important context (no behaviour instructions here): example: At this place apples are purple.
        messages.append({"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-04-10"})
        question = {}
        question['role'] = 'user'
        question['content'] = user_input
        messages.append(question)
        try:
            # Generate images using DALL-E
            response = openai.Image.create(model="image-alpha-001",prompt=user_input,n=1,size="512x512")
            print(response)
            # content = response.choices[0].message["content"]
            # Generate images using DALL-E
            content = response.data[0]['url']
            # count_tokens = response['usage']['total_tokens']
        except Exception as e:
            # monitor exception using Rollbar
            rollbar.report_exc_info()
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)
        except RateLimitError:
            content = "You have hit your assigned rate limit. "
        except AuthenticationError:
            content = "Your API key or token was invalid, expired, or revoked."
        except Timeout :
            content = "Request timed out."
        except APIError :
            content = "The server is experiencing a high volume of requests. Please try again later. Issue on our side."
        except APIConnectionError:  
            content = " Issue connecting to our services."
        except InvalidRequestError :
            content = "Your request was malformed or missing some required parameters, such as a token or an input."
        except openai.error.PermissionError :
            content = "Your API key or token does not have the required scope or role to perform the requested action."
        except ServiceUnavailableError :
            content = "Issue on our servers."
        

        return jsonify(content=content), 200
   
   return render_template('chatbotimage.html', page="Chatbot project",**locals())


def save_response_token(token_val):
    try:
       create_dt=datetime.datetime.now()
       conn = get_db_connection() 
       cur = conn.cursor()  
       cur.execute("INSERT into value_token (token_val,create_dt) values (?,?)",(token_val,create_dt))  
       conn.commit()  
    except sqlite3.Error as er:
        print('SQLite error: %s' % (' '.join(er.args)))
        print("Exception class is: ", er.__class__)
        print('SQLite traceback: ')
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:  
        conn.rollback()  
        print("db error")
    finally:  
        conn.close() 
        print("closed!!!")

def save_system_instruction(content):
    try:
       create_dt=datetime.datetime.now()
       conn = get_db_connection() 
       cur = conn.cursor()  
       cur.execute("INSERT into system_instruction (content,create_dt) values (?,?)",(content,create_dt))  
       conn.commit()  
    except sqlite3.Error as er:
        print('SQLite error: %s' % (' '.join(er.args)))
        print("Exception class is: ", er.__class__)
        print('SQLite traceback: ')
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:  
        conn.rollback()  
        print("db error")
    finally:  
        conn.close() 
        print("closed!!!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888',debug=True)