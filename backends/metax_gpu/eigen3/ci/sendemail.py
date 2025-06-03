#!/usr/bin/env python3

import smtplib
from email.message import EmailMessage
from email.headerregistry import Address
from junitparser import JUnitXml as Jx
from pathlib import Path
from base64 import b64decode
import os
import sys
import subprocess

class Summary:
    def __init__(self, title):
        self.tests = 0
        self.failures = 0
        self.errors = 0
        self.disabled = 0
        self.time = 0
        self.title = title
    def html(self):
       passed = self.tests - self.disabled - self.errors - self.failures
       return """\
<dl>
  <dt>{}</dt>
  <dd class="percentage percentage-{}">
    <span class="text"> Passed: {}% </span>
  </dd>
  <dd class="percentage percentage-{}">
    <span class="text"> failures: {}% </span>
  </dd>
  <dd class="percentage percentage-{}">
    <span class="text"> errors: {}% </span>
  </dd>
  <dd class="percentage percentage-{}">
    <span class="text"> disabled: {}% </span>
  </dd>
</dl>
""".format(self.title,
          float(passed / self.tests * 100), passed,
          float(self.failures / self.tests * 100), self.failures,
          float(self.errors / self.tests * 100), self.errors,
          float(self.disabled / self.tests * 100), self.disabled)

    def html_simple(self):
       passed = self.tests - self.disabled - self.errors - self.failures
       return """
       <table>
       <tr><th>Passed</th><th>Failures</th><th>Errors</th><th>Disabled</th></tr>
       <tr><td style="color:green">{:.2f}%({}),</td><td style="color:red">{:.2f}%({}),</td><td>{:.2f}%({}),</td><td style="color:gray">{:.2f}%({})</td></tr>
       </table>
       """.format(
          float(passed / self.tests * 100), passed,
          float(self.failures / self.tests * 100), self.failures,
          float(self.errors / self.tests * 100), self.errors,
          float(self.disabled / self.tests * 100), self.disabled)



def junit_xml_summary(summary, xml):
    result = Jx.fromfile(str(xml))
    summary.tests += result.tests
    summary.failures += result.failures
    summary.errors += result.errors
    summary.disabled += result.skipped
    summary.time += result.time
    return

def junit_xml_to_html(xml,is_sucess = True, sucess_cnt = 0, sucess_cases_summary_num = 100):
    result = Jx.fromfile(str(xml))
    tbl = ""
    failed_num = 0
    for testsuite in result:
        for testcase in testsuite:
            if is_sucess == True and len(testcase.result) == 0:
                if sucess_cnt>=sucess_cases_summary_num:
                  break
                tr = """\
                <tr><td>{}</td> <td>{}</td><td>{}</td><td>{}</td></tr>
                """.format(testcase.classname, testcase.name, testcase.time, "passed");
                sucess_cnt = sucess_cnt + 1
                tbl += tr

            elif is_sucess==False and len(testcase.result) != 0:
                tr = """\
                <tr class="fail"><td>{}</td> <td>{}</td><td>{}</td><td>{}</td></tr>
                """.format(testcase.classname, testcase.name, testcase.time, "failed");
                tbl += tr
                failed_num = failed_num + 1
    return tbl, sucess_cnt, failed_num

def xmls_to_html(summary, xmls, sucess_cases_summary_num = 100):

  content = []
  html_table = ""
  failed_num = 0
  status = 0
  for xml in xmls:
      filesize = os.path.getsize(xml)
      if int(filesize) == 0:
        status = status + 1
        continue
      with open(xml, "r") as fp:
          content.append(fp.read())
          junit_xml_summary(summary, xml)
          html_table_sub, _, cnt = junit_xml_to_html(xml, is_sucess=False)
          html_table = html_table + html_table_sub
          failed_num = failed_num + cnt

  cnt=0
  for xml in xmls:
      filesize = os.path.getsize(xml)
      if int(filesize) == 0:
        continue
      with open(xml,"r") as fp:
        html_table_sub, cnt, _ = junit_xml_to_html(xml, is_sucess = True,
                    sucess_cnt = cnt, sucess_cases_summary_num = sucess_cases_summary_num)
        html_table = html_table + html_table_sub
  html_header = """\
<p>tests {} failures {} disabled {} time(s) {}</p>
<table id="custom">
<tr>
<th>classname</th>
<th>name</th>
<th>time(s)</th>
<th>failures</th>
</tr>
""".format(summary.tests, summary.failures, summary.disabled, summary.time)
  html = html_header + html_table + "</table>"

  if failed_num != 0 or cnt == 0:
    status = status + 1
  return html, content, status


"""
dl {
  display: flex;
  background-color: white;
  flex-direction: column;
  width: 100%;
  max-width: 700px;
  position: relative;
  padding: 20px;
}
dt {
  align-self: flex-start;
  width: 100%;
  font-weight: 700;
  display: block;
  text-align: center;
  font-size: 1.2em;
  font-weight: 700;
  margin-bottom: 20px;
  margin-left: 130px;
}
.text {
  font-weight: 600;
  display: flex;
  align-items: center;
  height: 40px;
  width: 130px;
  background-color: white;
  position: absolute;
  left: 0;
  justify-content: flex-end;
}
.percentage {
  font-size: 0.8em;
  line-height: 1;
  text-transform: uppercase;
  width: 100%;
  height: 40px;
  margin-left: 130px;
  background: repeating-linear-gradient(to right, #ddd, #ddd 1px, #fff 1px, #fff 5%);
}
.percentage:after {
  content: "";
  display: block;
  background-color: #3d9970;
  width: 50px;
  margin-bottom: 10px;
  height: 90%;
  position: relative;
  top: 50%;
  transform: translateY(-50%);
  transition: background-color 0.3s ease;
  cursor: pointer;
}
.percentage:hover:after, .percentage:focus:after { background-color: #aaa; }
.percentage-0:after{width:0%}
.percentage-1:after{width:1%}
.percentage-2:after{width:2%}
.percentage-3:after{width:3%}
.percentage-4:after{width:4%}
.percentage-5:after{width:5%}
.percentage-6:after{width:6%}
.percentage-7:after{width:7%}
.percentage-8:after{width:8%}
.percentage-9:after{width:9%}
.percentage-10:after{width:10%}
.percentage-11:after{width:11%}
.percentage-12:after{width:12%}
.percentage-13:after{width:13%}
.percentage-14:after{width:14%}
.percentage-15:after{width:15%}
.percentage-16:after{width:16%}
.percentage-17:after{width:17%}
.percentage-18:after{width:18%}
.percentage-19:after{width:19%}
.percentage-20:after{width:20%}
.percentage-21:after{width:21%}
.percentage-22:after{width:22%}
.percentage-23:after{width:23%}
.percentage-24:after{width:24%}
.percentage-25:after{width:25%}
.percentage-26:after{width:26%}
.percentage-27:after{width:27%}
.percentage-28:after{width:28%}
.percentage-29:after{width:29%}
.percentage-30:after{width:30%}
.percentage-31:after{width:31%}
.percentage-32:after{width:32%}
.percentage-33:after{width:33%}
.percentage-34:after{width:34%}
.percentage-35:after{width:35%}
.percentage-36:after{width:36%}
.percentage-37:after{width:37%}
.percentage-38:after{width:38%}
.percentage-39:after{width:39%}
.percentage-40:after{width:40%}
.percentage-41:after{width:41%}
.percentage-42:after{width:42%}
.percentage-43:after{width:43%}
.percentage-44:after{width:44%}
.percentage-45:after{width:45%}
.percentage-46:after{width:46%}
.percentage-47:after{width:47%}
.percentage-48:after{width:48%}
.percentage-49:after{width:49%}
.percentage-50:after{width:50%}
.percentage-51:after{width:51%}
.percentage-52:after{width:52%}
.percentage-53:after{width:53%}
.percentage-54:after{width:54%}
.percentage-55:after{width:55%}
.percentage-56:after{width:56%}
.percentage-57:after{width:57%}
.percentage-58:after{width:58%}
.percentage-59:after{width:59%}
.percentage-60:after{width:60%}
.percentage-61:after{width:61%}
.percentage-62:after{width:62%}
.percentage-63:after{width:63%}
.percentage-64:after{width:64%}
.percentage-65:after{width:65%}
.percentage-66:after{width:66%}
.percentage-67:after{width:67%}
.percentage-68:after{width:68%}
.percentage-69:after{width:69%}
.percentage-70:after{width:70%}
.percentage-71:after{width:71%}
.percentage-72:after{width:72%}
.percentage-73:after{width:73%}
.percentage-74:after{width:74%}
.percentage-75:after{width:75%}
.percentage-76:after{width:76%}
.percentage-77:after{width:77%}
.percentage-78:after{width:78%}
.percentage-79:after{width:79%}
.percentage-80:after{width:80%}
.percentage-81:after{width:81%}
.percentage-82:after{width:82%}
.percentage-83:after{width:83%}
.percentage-84:after{width:84%}
.percentage-85:after{width:85%}
.percentage-86:after{width:86%}
.percentage-87:after{width:87%}
.percentage-88:after{width:88%}
.percentage-89:after{width:89%}
.percentage-90:after{width:90%}
.percentage-91:after{width:91%}
.percentage-92:after{width:92%}
.percentage-93:after{width:93%}
.percentage-94:after{width:94%}
.percentage-95:after{width:95%}
.percentage-96:after{width:96%}
.percentage-97:after{width:97%}
.percentage-98:after{width:98%}
.percentage-99:after{width:99%}
.percentage-100:after{width:100%}
"""

table_css= """\
#custom {
  font-family: Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}
#custom td, #custom th {
  border: 1px solid #ddd;
  padding: 2px;
}
#custom tr:hover {background-color: #ddd;}
#custom tr:nth-child(even) {background-color: #f2f2f2;}
tr.fail {background-color: red;}
#custom th {
  text-align: left;
  background-color: #04AA6D;
  color: white;
  padding: 10px;
}"""

class GitLog:
    def __init__(self, path):
        self._path = Path(path)
        if not self._path.exists():
            print("Error: path {} not exists".format(path))

    def __str__(self):
        if not self._path.exists():
            return ""
        return subprocess.check_output("cd {} && git log --oneline | head -n 5".format(self._path.absolute()),
            shell=True).decode('ascii');
    def html(self):
        if not self._path.exists():
            return ""
        lines = self.__str__().strip()
        htm = '<h3>{}</h3><ul id="custom">'.format(self._path.absolute().stem)
        for line in lines.split("\n"):
            htm += "<li>{}</li>".format(line)
        htm += "</ul>"
        return htm

def get_env(env_name):
    if env_name in os.environ:
        return os.environ[env_name]
    else:
        return ""


def send_junit_xml(xmllist,sucess_cases_summary_num = 10,
                   from_=Address("ACL_EIGEN_CI", "jenkins_mafe_bot", "metax-tech.com")):
    msg = EmailMessage()
    jenkins_build_url = get_env("BUILD_URL")
    jenkins_build_htm = """<p id="custom">JenkinsBuildUrl: <a href="{}">{}</a></p>""".format(
                        jenkins_build_url, jenkins_build_url)
    gerrit_change_url = get_env("GERRIT_CHANGE_URL")
    gerrit_change_htm = "" if len(gerrit_change_url) == 0 else """<p id="custom">GerritChangeUrl: <a href="{}">{}</a></p>""".format(
                                gerrit_change_url, gerrit_change_url)
    sucess_num_htm= """<p id="custom">Only Show Max {} Sucessful cases and All Failured cases </p>""".format(
                                sucess_cases_summary_num)
    gerrit_change_htm=gerrit_change_htm+sucess_num_htm
    mail_subject = get_env("SEND_MAIL_SUBJECT")
    mail_to = get_env("SEND_MAIL_TO")

    summary = Summary(mail_subject)

    msg["From"] = from_
    msg["To"] = "xuewei.bei@metax-tech.com" if len(mail_to) == 0 else mail_to
    msg["Subject"] = "mcEigen_checkin_result" if len(mail_subject) == 0 else mail_subject

    html_table, content,status = xmls_to_html(summary, xmllist, sucess_cases_summary_num)
    contents = "\n".join(content)

    msg.set_content(contents)
    msg.add_alternative("""\
<!DOCTYPE html>
<html>
<head><style>{}</style></head>
<body>
{}
{}
{}
<hr />
{}
</body>
</html>""".format(table_css,
                  summary.html_simple(),
                  gerrit_change_htm, jenkins_build_htm,
                  html_table),
                subtype='html')
    try:
      s=smtplib.SMTP("mail.metax-tech.com", timeout=300)
      s.send_message(msg)
    except:
        print("Cannot Send Email Successfully")
    return status


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: {} xmlpath [xmlpath2 ...]".format(sys.argv[0]))
        print("\t export SEND_MAIL_TO=a,b,c")
        print("\t export SEND_MAIL_SUBJECT=abc")
        exit(1)

    xmls = []
    for _path in sys.argv[1:]:
        xmls += sorted(
            Path(_path).glob("**/*.xml")
        )

    status=0
    if len(xmls) > 0:
        status=status+send_junit_xml(xmls,sucess_cases_summary_num=int(sys.argv[2]))
    else:
        status=1
    sys.exit(status)

