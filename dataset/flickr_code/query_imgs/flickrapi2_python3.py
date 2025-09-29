# -*- coding: utf-8 -*-
#!/usr/bin/python
#
# Flickr API implementation
#
# Inspired largely by Michele Campeotto's flickrclient and Aaron Swartz'
# xmltramp... but I wanted to get a better idea of how python worked in
# those regards, so I mostly worked those components out for myself.
#
# http://micampe.it/things/flickrclient
# http://www.aaronsw.com/2002/xmltramp/
#
# Release 1: initial release
# Release 2: added upload functionality
# Release 3: code cleanup, convert to doc strings
# Release 4: better permission support
# Release 5: converted into fuller-featured "flickrapi"
# Release 6: fix upload sig bug (thanks Deepak Jois), encode test output
# Release 7: fix path construction, Manish Rai Jain's improvements, exceptions
# Release 8: change API endpoint to "api.flickr.com"
#
# Work by (or inspired by) Manish Rai Jain <manishrjain@gmail.com>:
#
#    improved error reporting, proper multipart MIME boundary creation,
#    use of urllib2 to allow uploads through a proxy, upload accepts
#    raw data as well as a filename
#
# Copyright 2005 Brian "Beej Jorgensen" Hall <beej@beej.us>
#
#    This work is licensed under the Creative Commons
#    Attribution License.  To view a copy of this license,
#    visit http://creativecommons.org/licenses/by/2.5/ or send
#    a letter to Creative Commons, 543 Howard Street, 5th
#    Floor, San Francisco, California, 94105, USA.
#
# This license says that I must be credited for any derivative works.
# You do not need to credit me to simply use the FlickrAPI classes in
# your Python scripts--you only need to credit me if you're taking this
# FlickrAPI class and modifying it or redistributing it.
#
# Previous versions of this API were granted to the public domain.
# You're free to use those as you please.
#
# Beej Jorgensen, Maintainer, November 2005
# beej@beej.us
#

import sys
import hashlib
import urllib.parse
import urllib.request
import os.path
import xml.dom.minidom
import xml.dom  # for xml.dom.Node
from typing import Any, Dict
import uuid


########################################################################
# Exceptions
########################################################################

class UploadException(Exception):
    pass


########################################################################
# XML functionality
########################################################################

#-----------------------------------------------------------------------
class XMLNode:
    """XMLNode -- generic class for holding an XML node

    xmlStr = \"\"\"<xml foo="32">
    <name bar="10">Name0</name>
    <name bar="11" baz="12">Name1</name>
    </xml>\"\"\"

    f = XMLNode.parseXML(xmlStr)

    print(f.elementName)              # xml
    print(f['foo'])                   # 32
    print(f.name)                     # [<name XMLNode>, <name XMLNode>]
    print(f.name[0].elementName)      # name
    print(f.name[0]["bar"])           # 10
    print(f.name[0].elementText)      # Name0
    print(f.name[1].elementName)      # name
    print(f.name[1]["bar"])           # 11
    print(f.name[1]["baz"])           # 12

    """

    def __init__(self):
        """Construct an empty XML node."""
        self.elementName = ""
        self.elementText = ""
        self.attrib: Dict[str, str] = {}
        self.xml = ""

    def __setitem__(self, key, item):
        """Store a node's attribute in the attrib hash."""
        self.attrib[key] = item

    def __getitem__(self, key):
        """Retrieve a node's attribute from the attrib hash."""
        try:
            return self.attrib[key]
        except Exception:
            return "null"

    @classmethod
    def parseXML(cls, xmlStr, storeXML=False):
        """Convert an XML string into a nice instance tree of XMLNodes.

        xmlStr -- the XML to parse
        storeXML -- if True, stores the XML string in the root XMLNode.xml
        """

        def __parseXMLElement(element, thisNode):
            """Recursive call to process this XMLNode."""
            thisNode.elementName = element.nodeName

            # add element attributes as attributes to this node
            if element.attributes:
                for i in range(element.attributes.length):
                    an = element.attributes.item(i)
                    thisNode[an.name] = an.nodeValue

            for a in element.childNodes:
                if a.nodeType == xml.dom.Node.ELEMENT_NODE:

                    child = XMLNode()
                    try:
                        lst = getattr(thisNode, a.nodeName)
                    except AttributeError:
                        setattr(thisNode, a.nodeName, [])

                    # add the child node as an attrib to this node
                    lst = getattr(thisNode, a.nodeName)
                    lst.append(child)

                    __parseXMLElement(a, child)

                elif a.nodeType == xml.dom.Node.TEXT_NODE:
                    thisNode.elementText += a.nodeValue or ""

            return thisNode

        dom = xml.dom.minidom.parseString(xmlStr)

        # get the root
        rootNode = XMLNode()
        if storeXML:
            rootNode.xml = xmlStr

        return __parseXMLElement(dom.firstChild, rootNode)


########################################################################
# Flickr functionality
########################################################################

#-----------------------------------------------------------------------
class FlickrAPI:
    """Encapsulated flickr functionality.

    Example usage:

      flickr = FlickrAPI(flickrAPIKey, flickrSecret)
      rsp = flickr.auth_checkToken(api_key=flickrAPIKey, auth_token=token)

    """
    flickrHost = "api.flickr.com"
    flickrRESTForm = "/services/rest/"
    flickrAuthForm = "/services/auth/"
    flickrUploadForm = "/services/upload/"

    #-------------------------------------------------------------------
    def __init__(self, apiKey, secret):
        """Construct a new FlickrAPI instance for a given API key and secret."""
        self.apiKey = apiKey
        self.secret = secret

        self.__handlerCache: Dict[str, Any] = {}

    #-------------------------------------------------------------------
    def __sign(self, data: Dict[str, str]) -> str:
        """Calculate the flickr signature for a set of params.

        data -- a hash of all the params and values to be hashed, e.g.
                {"api_key":"AAAA", "auth_token":"TTTT"}
        """
        dataName = self.secret
        keys = sorted(data.keys())
        for a in keys:
            dataName += (a + str(data[a]))
        h = hashlib.md5()
        h.update(dataName.encode("utf-8"))
        return h.hexdigest()

    #-------------------------------------------------------------------
    def __getattr__(self, method, **arg):
        """Handle all the flickr API calls.

        General handler for methods not defined; assumes they are
        flickr methods. Converts them to method= parameter, posts, parses.

        example usage:

            flickr.auth_getFrob(api_key="AAAAAA")
            rsp = flickr.favorites_getList(api_key=flickrAPIKey,
                auth_token=token)
        """

        if method not in self.__handlerCache:
            def handler(_self=self, _method=method, **arg):
                flickr_method = "flickr." + _method.replace("_", ".")
                url = "https://" + FlickrAPI.flickrHost + FlickrAPI.flickrRESTForm
                arg["method"] = flickr_method
                qs = urllib.parse.urlencode(arg)
                qs += "&api_sig=" + _self.__sign(arg)
                req = urllib.request.Request(
                    url + "?" + qs,
                    headers={
                        "User-Agent": "flickrapi2-py3",
                        "Accept": "application/xml"
                    }
                )
                with urllib.request.urlopen(req, timeout=60) as f:
                    data = f.read().decode("utf-8")
                return XMLNode.parseXML(data, True)

            self.__handlerCache[method] = handler

        return self.__handlerCache[method]

    #-------------------------------------------------------------------
    def __getAuthURL(self, perms, frob):
        """Return the authorization URL to get a token.

        perms -- "read", "write", or "delete"
        frob -- picked up from an earlier call to FlickrAPI.auth_getFrob()
        """

        data = {"api_key": self.apiKey, "frob": frob, "perms": perms}
        data["api_sig"] = self.__sign(data)
        return "https://%s%s?%s" % (FlickrAPI.flickrHost,
            FlickrAPI.flickrAuthForm, urllib.parse.urlencode(data))

    #-------------------------------------------------------------------
    def upload(self, filename=None, jpegData=None, **arg):
        """Upload a file to flickr.

        Be extra careful you spell the parameters correctly, or you will
        get a rather cryptic "Invalid Signature" error on the upload!

        Supported parameters:

        One of filename or jpegData must be specified by name when
        calling this method:

        filename -- name of a file to upload
        jpegData -- array of jpeg data to upload

        api_key
        auth_token
        title
        description
        tags -- space-delimited list of tags, "tag1 tag2 tag3"
        is_public -- "1" or "0"
        is_friend -- "1" or "0"
        is_family -- "1" or "0"
        """

        if (filename is None and jpegData is None) or \
           (filename is not None and jpegData is not None):
            raise UploadException("filename OR jpegData must be specified")

        # verify key names
        for a in arg.keys():
            if a not in ("api_key", "auth_token", "title", "description", "tags",
                         "is_public", "is_friend", "is_family"):
                sys.stderr.write(
                    'FlickrAPI: warning: unknown parameter "%s" sent to FlickrAPI.upload\n' % (a)
                )

        arg["api_sig"] = self.__sign(arg)
        url = "http://" + FlickrAPI.flickrHost + FlickrAPI.flickrUploadForm

        # construct POST data
        boundary = uuid.uuid4().hex  # replacement for mimetools.choose_boundary()
        body = ""

        # required params
        for a in ('api_key', 'auth_token', 'api_sig'):
            if a in arg:
                body += "--%s\r\n" % (boundary)
                body += 'Content-Disposition: form-data; name="%s"\r\n\r\n' % a
                body += "%s\r\n" % (arg[a])

        # optional params
        for a in ('title', 'description', 'tags', 'is_public', 'is_friend', 'is_family'):
            if a in arg:
                body += "--%s\r\n" % (boundary)
                body += 'Content-Disposition: form-data; name="%s"\r\n\r\n' % a
                body += "%s\r\n" % (arg[a])

        body += "--%s\r\n" % (boundary)
        body += 'Content-Disposition: form-data; name="photo";'
        body += ' filename="%s"\r\n' % filename
        body += "Content-Type: image/jpeg\r\n\r\n"

        if filename is not None:
            with open(filename, "rb") as fp:
                data = fp.read()
        else:
            data = jpegData

        postData = body.encode("utf-8") + data + ("--%s--" % (boundary)).encode("utf-8")

        req = urllib.request.Request(url, data=postData)
        req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
        with urllib.request.urlopen(req) as response:
            rspXML = response.read().decode("utf-8")

        return XMLNode.parseXML(rspXML)

    #-----------------------------------------------------------------------
    @classmethod
    def testFailure(cls, rsp, exit=True):
        """Exit app if the rsp XMLNode indicates failure."""
        if rsp['stat'] == "fail":
            sys.stderr.write("%s\n" % (cls.getPrintableError(rsp)))
            if exit:
                sys.exit(1)

    #-----------------------------------------------------------------------
    @classmethod
    def getPrintableError(cls, rsp):
        """Return a printed error message string."""
        return "%s: error %s: %s" % (rsp.elementName,
            cls.getRspErrorCode(rsp), cls.getRspErrorMsg(rsp))

    #-----------------------------------------------------------------------
    @classmethod
    def getRspErrorCode(cls, rsp):
        """Return the error code of a response, or 0 if no error."""
        if rsp['stat'] == "fail":
            return rsp.err[0]['code']
        return 0

    #-----------------------------------------------------------------------
    @classmethod
    def getRspErrorMsg(cls, rsp):
        """Return the error message of a response, or "Success" if no error."""
        if rsp['stat'] == "fail":
            return rsp.err[0]['msg']
        return "Success"

    #-----------------------------------------------------------------------
    def __getCachedTokenPath(self):
        """Return the directory holding the app data."""
        return os.path.expanduser(os.path.sep.join(["~", ".flickr",
            self.apiKey]))

    #-----------------------------------------------------------------------
    def __getCachedTokenFilename(self):
        """Return the full pathname of the cached token file."""
        return os.path.sep.join([self.__getCachedTokenPath(), "auth.xml"])

    #-----------------------------------------------------------------------
    def __getCachedToken(self):
        """Read and return a cached token, or None if not found.

        The token is read from the cached token file, which is basically the
        entire RSP response containing the auth element.
        """
        try:
            with open(self.__getCachedTokenFilename(), "r", encoding="utf-8") as f:
                data = f.read()
            rsp = XMLNode.parseXML(data)
            return rsp.auth[0].token[0].elementText
        except IOError:
            return None

    #-----------------------------------------------------------------------
    def __setCachedToken(self, xml):
        """Cache a token for later use.

        The cached tag is stored by simply saving the entire RSP response
        containing the auth element.
        """
        path = self.__getCachedTokenPath()
        if not os.path.exists(path):
            os.makedirs(path)

        with open(self.__getCachedTokenFilename(), "w", encoding="utf-8") as f:
            f.write(xml)

    #-----------------------------------------------------------------------
    def getToken(self, perms="read", browser="lynx"):
        """Get a token either from the cache, or make a new one from the frob."""
        # see if we have a saved token
        token = self.__getCachedToken()

        # see if it's valid
        if token is not None:
            rsp = self.auth_checkToken(api_key=self.apiKey, auth_token=token)
            if rsp['stat'] != "ok":
                token = None
            else:
                # see if we have enough permissions
                tokenPerms = rsp.auth[0].perms[0].elementText
                if tokenPerms == "read" and perms != "read":
                    token = None
                elif tokenPerms == "write" and perms == "delete":
                    token = None

        # get a new token if we need one
        if token is None:
            # get the frob
            rsp = self.auth_getFrob(api_key=self.apiKey)
            self.testFailure(rsp)

            frob = rsp.frob[0].elementText

            # validate online
            os.system("%s '%s'" % (browser, self.__getAuthURL(perms, frob)))

            # get a token
            rsp = self.auth_getToken(api_key=self.apiKey, frob=frob)
            self.testFailure(rsp)

            token = rsp.auth[0].token[0].elementText

            # store the auth info for next time
            self.__setCachedToken(rsp.xml)

        return token


########################################################################
# App functionality (demo)
########################################################################

def main(argv):
    # flickr auth information (demo only): REPLACE with your keys to test main()
    flickrAPIKey = "7b679f94b19629fd157bc71bb1f10b71"
    flickrSecret = "973fe554fb3408e2"

    # make a new FlickrAPI instance
    fapi = FlickrAPI(flickrAPIKey, flickrSecret)

    # do the whole whatever-it-takes to get a valid token:
    token = fapi.getToken(browser="firefox")

    # get my favorites
    rsp = fapi.favorites_getList(api_key=flickrAPIKey, auth_token=token)
    fapi.testFailure(rsp)

    # and print them
    for a in rsp.photos[0].photo:
        title = a['title']
        try:
            title = title.encode("ascii", "replace").decode("ascii", "replace")
        except Exception:
            pass
        print("%10s: %s" % (a['id'], title))

    return 0

# run the main if we're not being imported:
if __name__ == "__main__":
    sys.exit(main(sys.argv))
