import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv
import base64
import io
import tempfile
import shutil

# 加载环境变量
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path, override=True)  # 添加 override=True 确保覆盖任何现有的环境变量

# 语言配置
TRANSLATIONS = {
    "zh": {
        "page_title": "基金申请政策智能问答系统",
        "page_icon_text": "📚",
        "system_config": "系统配置",
        "upload_pdf": "上传PDF文档",
        "chunk_size_text": "设置文档分块大小",
        "top_k_text": "设置检索结果数量(Top K)",
        "uploaded_files": "已上传的文件：",
        "reset_button": "重置",
        "welcome_message": "您好！请先上传PDF文档,然后我可以帮您解答相关问题。",
        "preview_title": "政策文件预览",
        "qa_title": "智能问答",
        "input_placeholder": "请输入您的问题：",
        "submit_button": "提交问题",
        "upload_first": "请先上传PDF文档",
        "processing_error": "文档处理失败",
        "latest_answer": "最新回答",
        "reference_basis": "条规依据",
        "related_basis": "相关依据",
        "source_info": "来源信息",
        "file_text": "文件",
        "page_text": "页码",
        "chat_history": "查看聊天历史",
        "question_prefix": "问题",
        "answer_prefix": "回答",
        "loading_text": "正在加载并索引文档，请稍候...",
        "reference_answer": "参考答案",
        "policy_source": "政策来源",
        "page_number": "所在页数",
        "page_prefix": "第",
        "page_suffix": "页",
        "pdf_display_error": "PDF显示出错",
        "doc_process_error": "处理问题时出错"
    },
    "en": {
        "page_title": "Intelligent Q&A System for Fund Application Policies",
        "page_icon_text": "📚",
        "system_config": "System Configuration",
        "upload_pdf": "Upload PDF Documents",
        "chunk_size_text": "Set Document Chunk Size",
        "top_k_text": "Set Number of Retrieved Results (Top K)",
        "uploaded_files": "Uploaded Files:",
        "reset_button": "Reset",
        "welcome_message": "Hello! Please upload PDF documents first, then I can help answer your questions.",
        "preview_title": "Policy Document Preview",
        "qa_title": "Intelligent Q&A",
        "input_placeholder": "Enter your question:",
        "submit_button": "Submit Question",
        "upload_first": "Please upload PDF documents first",
        "processing_error": "Document processing failed",
        "latest_answer": "Latest Answer",
        "reference_basis": "Reference Basis",
        "related_basis": "Related Basis",
        "source_info": "Source Information",
        "file_text": "File",
        "page_text": "Page",
        "chat_history": "View Chat History",
        "question_prefix": "Question",
        "answer_prefix": "Answer",
        "loading_text": "Loading and indexing documents, please wait...",
        "reference_answer": "Reference Answer",
        "policy_source": "Policy Source",
        "page_number": "Page Number",
        "page_prefix": "Page",
        "page_suffix": "",
        "pdf_display_error": "PDF display error",
        "doc_process_error": "Error processing question"
    }
}

# 初始化语言设置
if 'language' not in st.session_state:
    st.session_state.language = "zh"

# 获取翻译文本的函数
def get_text(key):
    return TRANSLATIONS[st.session_state.language][key]

# 页面配置
st.set_page_config(
    page_title=get_text("page_title"),
    page_icon=get_text("page_icon_text"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# OpenAI配置
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')

if not api_key:
    st.error("请设置 OPENAI_API_KEY 环境变量")
    st.stop()

# 确保环境变量被正确设置
os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = api_base if api_base else "https://api.xty.app/v1"

# 在页面配置后添加临时文件夹管理
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    
# 清理临时文件夹的函数
def cleanup_temp_dir():
    if 'temp_dir' in st.session_state:
        shutil.rmtree(st.session_state.temp_dir)
        del st.session_state.temp_dir

# PDF显示函数
def display_pdf(pdf_content, page_number=1):
    base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
    pdf_display = f'''
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}#page={page_number}&view=Fit"
            width="700"
            height="1000"
            type="application/pdf"
            style="border: none; margin: 0; padding: 0;"
        ></iframe>
    '''
    st.markdown(
        """
        <style>
        .stMarkdown {
            padding: 0 !important;
            margin: 0 !important;
        }
        iframe {
            display: block;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(pdf_display, unsafe_allow_html=True)

# 加载和索引数据
@st.cache_resource(show_spinner=False)
def load_data(pdf_files=None, chunk_size=1000):
    with st.spinner(get_text("loading_text")):
        try:
            Settings.embed_model = OpenAIEmbedding(
                api_base=os.getenv('OPENAI_API_BASE'),
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            all_docs = []
            
            if pdf_files:
                # 保存所有上传的PDF到临时文件夹
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(st.session_state.temp_dir, pdf_file.name)
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_file.getvalue())
                
                # 从临时文件夹读取所有文档
                reader = SimpleDirectoryReader(input_dir=st.session_state.temp_dir)
                all_docs = reader.load_data()
            else:
                # 使用相对路径加载默认文档
                default_docs_path = os.path.join(os.path.dirname(__file__), '..', 'data')
                reader = SimpleDirectoryReader(input_dir=default_docs_path, recursive=True)
                all_docs = reader.load_data()
            
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(
                    model="gpt-3.5-turbo", 
                    temperature=0.7,
                    api_base=os.getenv('OPENAI_API_BASE'),
                    api_key=os.getenv('OPENAI_API_KEY')
                ),
                chunk_size=chunk_size
            )
            index = VectorStoreIndex.from_documents(all_docs, service_context=service_context)
            return index, [file.getvalue() for file in pdf_files] if pdf_files else None
            
        except Exception as e:
            st.error(f"{get_text('processing_error')}: {str(e)}")
            return None, None

# 修改状态控制
if 'current_chat_pdf' not in st.session_state:
    st.session_state.current_chat_pdf = None
if 'is_chatting' not in st.session_state:
    st.session_state.is_chatting = False

# 侧边栏配置
with st.sidebar:
    st.title(get_text("system_config"))
    
    # 添加语言选择
    selected_language = st.selectbox(
        "Language/语言",
        options=["中文", "English"],
        index=0 if st.session_state.language == "zh" else 1
    )
    
    # 更新语言设置
    if selected_language == "中文" and st.session_state.language != "zh":
        st.session_state.language = "zh"
        st.rerun()
    elif selected_language == "English" and st.session_state.language != "en":
        st.session_state.language = "en"
        st.rerun()
    
    uploaded_files = st.file_uploader(get_text("upload_pdf"), type=['pdf'], accept_multiple_files=True)
    chunk_size = st.slider(get_text("chunk_size_text"), 500, 2000, 1000)
    top_k = st.slider(get_text("top_k_text"), 1, 10, 3)
    
    # 显示已上传的文件列表
    if uploaded_files:
        st.write(get_text("uploaded_files"))
        for file in uploaded_files:
            st.write(f"- {file.name}")
    
    if st.button(get_text("reset_button")):
        cleanup_temp_dir()
        st.session_state.temp_dir = tempfile.mkdtemp()
        if 'chat_engine' in st.session_state:
            del st.session_state.chat_engine
        st.session_state.is_chatting = False
        st.session_state.current_chat_pdf = None
        st.session_state.messages = [
            {"role": "assistant", "content": get_text("welcome_message")}
        ]
        st.rerun()

# 主界面标题
st.title(get_text("page_title"))

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": get_text("welcome_message")}
    ]

# 分列布局
col1, col2 = st.columns([0.45, 0.55])

# PDF显示部分
with col1:
    st.subheader(get_text("preview_title"))
    
    # 初始状态显示NSFC图片
    if not st.session_state.is_chatting:
        image_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'logo.jpg')
        st.image(image_path)
    
    # 显示检索到的PDF
    elif st.session_state.current_chat_pdf:
        try:
            st.session_state.current_chat_pdf.seek(0)
            pdf_content = st.session_state.current_chat_pdf.read()
            display_pdf(pdf_content)
        except Exception as e:
            st.error(f"{get_text('pdf_display_error')}: {str(e)}")
            st.session_state.is_chatting = False
            st.session_state.current_chat_pdf = None
            if hasattr(st.session_state, 'current_response'):
                del st.session_state.current_response
            st.rerun()

# 问答界面
with col2:
    st.subheader(get_text("qa_title"))
    prompt = st.text_input(get_text("input_placeholder"))
    
    if st.button(get_text("submit_button"), use_container_width=True):
        if not uploaded_files:
            st.error(get_text("upload_first"))
            st.stop()
        
        if "chat_engine" not in st.session_state:
            index, _ = load_data(uploaded_files, chunk_size)
            if index:
                st.session_state.chat_engine = index.as_chat_engine(
                    chat_mode="condense_question", 
                    verbose=True,
                    similarity_top_k=top_k
                )
            else:
                st.error(get_text("processing_error"))
                st.stop()
        
        try:
            # 获取回答
            response = st.session_state.chat_engine.chat(prompt)
            
            # 更新当前PDF为检索到的第一个文档
            first_node = response.source_nodes[0].node
            file_name = first_node.metadata.get('file_name')
            for file in uploaded_files:
                if file.name == file_name:
                    st.session_state.current_chat_pdf = file
                    st.session_state.is_chatting = True
                    break
            
            # 保存到会话状态
            st.session_state.current_response = response
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 修改输出部分的代码
            output = f"**{get_text('reference_answer')}:** {response.response}\n"
            text_content = response.source_nodes[0].node.text
            cleaned_text = text_content.replace('\n', ' ')  # 先处理文本
            output += f"**{get_text('reference_basis')}:** {cleaned_text}\n"
            output += f"**{get_text('policy_source')}:** {response.source_nodes[0].node.metadata.get('file_name', '未知文件')}\n"
            output += f"**{get_text('page_number')}:** {get_text('page_prefix')}{response.source_nodes[0].node.metadata.get('page_label', '未知')}{get_text('page_suffix')}"
            
            st.session_state.messages.append({"role": "assistant", "content": output})
            
            # 强制重新渲染
            st.rerun()
            
        except Exception as e:
            st.error(f"{get_text('doc_process_error')}: {str(e)}")
    
    # 显示当前回答（如果有）
    if hasattr(st.session_state, 'current_response'):
        response = st.session_state.current_response
        
        # 显示最新的回答
        st.markdown(f"### {get_text('latest_answer')}")
        st.write(response.response)
        st.markdown("---")
        
        # 显示条规依据
        st.markdown(f"### {get_text('reference_basis')}")
        for i, source_node in enumerate(response.source_nodes):
            # 在显示条规依据部分
            with st.expander(f"{get_text('related_basis')} {i+1}", expanded=(i==0)):
                # 添加文本内容
                text = source_node.node.text
                cleaned_text = text.replace('\n', ' ')  # 先处理文本
                st.markdown(cleaned_text)
                
                # 添加分隔线
                st.markdown("---")
                
                # 添加来源信息，使用更清晰的格式
                file_name = source_node.node.metadata.get('file_name', '未知文件')
                page_label = source_node.node.metadata.get('page_label', '未知')
                source_info = f"""
                **{get_text('source_info')}**  
                📄 {get_text('file_text')}：{file_name}  
                📑 {get_text('page_text')}：{get_text('page_prefix')}{page_label}{get_text('page_suffix')}
                """
                st.markdown(source_info)

# 显示聊天历史
with st.expander(get_text("chat_history"), expanded=False):
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**{get_text('question_prefix')}：** {message['content']}")
        else:
            st.markdown(f"**{get_text('answer_prefix')}：** {message['content']}")
        st.markdown("---")