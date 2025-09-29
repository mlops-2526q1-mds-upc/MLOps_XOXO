# -*- coding: utf-8 -*-
#!/usr/bin/python

# Image querying script written by Tamara Berg,
# and extended heavily James Hays
#
# 9/26/2007 added dynamic timeslices to query more efficiently.
# 8/18/2008 added new fields and set maximum time slice.
# 8/19/2008 this is a much simpler function which gets ALL geotagged photos of
# sufficient accuracy.  No queries, no negative constraints.
# divides up the query results into multiple files
# 1/5/2009
# now uses date_taken instead of date_upload to get more diverse blocks of images
# 1/13/2009 - uses the original im2gps keywords, not as negative constraints though

import sys, string, math, time, socket
from flickrapi2_python3 import FlickrAPI
from datetime import datetime
import os

socket.setdefaulttimeout(30)  # 30 second time out on sockets before they throw
# an exception.  I've been having trouble with urllib.urlopen hanging in the
# flickr API.  This will show up as exceptions.IOError.

# the time out needs to be pretty long, it seems, because the flickr servers can be slow
# to respond to our big searches.

print(sys.argv)
if len(sys.argv) > 1:
    print("Reading queries from file " + sys.argv[1])
    query_file_name = sys.argv[1]  # 0 is the command name.
else:
    print("No command line arguments, reading queries from " + 'queries.txt')
    query_file_name = 'place_rec_queries.txt'
    #query_file_name = 'place_rec_queries_fall08.txt'

###########################################################################
# Modify this section to reflect your data and specific search
###########################################################################
# flickr auth information:
# change these to your flickr api keys and secret
flickrAPIKey = "7b679f94b19629fd157bc71bb1f10b71"   # <<--- PON AQUÍ TU API KEY
flickrSecret = "973fe554fb3408e2"  # <<--- Y AQUÍ TU API SECRET

query_file = open(query_file_name, 'r', encoding='utf-8')

# aggregate all of the positive and negative queries together.
pos_queries = []  # an empty list
neg_queries = ''  # a string
num_queries = 0

for line in query_file:
    if line[0] != '#' and len(line) > 2:  # line end character is 2 long?
        # print line[0:len(line)-2]
        if line[0] != '-':
            pos_queries = pos_queries + [line[0:len(line)-2]]
            num_queries = num_queries + 1
        if line[0] == '-':
            neg_queries = neg_queries + ' ' + line[0:len(line)-2]

query_file.close()
print('positive queries:  ')
print(pos_queries)
print('negative queries:  ' + neg_queries)
print('num_queries = ' + str(num_queries))

# make a new FlickrAPI instance
fapi = FlickrAPI(flickrAPIKey, flickrSecret)

for current_tag in range(0, num_queries):

    # change this to the location where you want to put your output file
    # out_file = open('/nfs/hn26/jhhays/download_scripts/search_results_geo_and_gps_siggraph/' + pos_queries[current_tag] + '.txt', 'w', encoding='utf-8')
    
    os.makedirs("search_results", exist_ok=True)
    out_file = open(os.path.join("search_results", pos_queries[current_tag] + ".txt"), "w", encoding="utf-8")

    ###########################################################################

    # form the query string.
    query_string = pos_queries[current_tag] + neg_queries
    print('\n\nquery_string is ' + query_string)
    total_images_queried = 0

    # number of seconds to skip per query
    #timeskip = 62899200 #two years
    #timeskip = 604800  #one week
    timeskip = 172800  #two days
    #timeskip = 86400 #one day
    #timeskip = 3600 #one hour
    #timeskip = 2257 #for resuming previous query

    #mintime = 1121832000 #from im2gps
    #mintime = 1167407788 # resume crash england
    #mintime = 1177828976 #resume crash japan
    #mintime = 1187753798 #resume crash greece
    mintime = 1171416400 # resume crash WashingtonDC
    maxtime = mintime + timeskip
    endtime = 1192165200  # 10/12/2007, at the end of im2gps queries

    # this is the desired number of photos in each block
    desired_photos = 250

    print(datetime.fromtimestamp(mintime))
    print(datetime.fromtimestamp(endtime))

    while (maxtime < endtime):

        # new approach - adjust maxtime until we get the desired number of images
        # within a block. We'll need to keep upper bounds and lower
        # lower bound is well defined (mintime), but upper bound is not. We can't
        # search all the way from endtime.

        lower_bound = mintime + 900  # lower bound OF the upper time limit. must be at least 15 minutes or zero results
        upper_bound = mintime + timeskip * 20  # upper bound of the upper time limit
        maxtime     = .95 * lower_bound + .05 * upper_bound

        print('\nBinary search on time range upper bound')
        print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
        print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

        keep_going = 6
        while keep_going > 0 and maxtime < endtime:
            try:
                rsp = fapi.photos_search(
                    api_key=flickrAPIKey,
                    ispublic="1",
                    media="photos",
                    per_page="250",
                    page="1",
                    has_geo="1",
                    text=query_string,
                    accuracy="6",
                    min_upload_date=str(mintime),
                    max_upload_date=str(maxtime)
                )
                time.sleep(1)
                FlickrAPI.testFailure(rsp)
                total_images = rsp.photos[0]['total']
                _ = float(total_images)  # sanity check

                print('\nnumimgs: ' + total_images)
                print('mintime: ' + str(mintime) + ' maxtime: ' + str(maxtime) +
                      ' timeskip: ' + str(maxtime - mintime))

                if int(total_images) > desired_photos:
                    print('too many photos in block, reducing maxtime')
                    upper_bound = maxtime
                    maxtime = (lower_bound + maxtime) / 2

                if int(total_images) < desired_photos:
                    print('too few photos in block, increasing maxtime')
                    lower_bound = maxtime
                    maxtime = (upper_bound + maxtime) / 2

                print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
                print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

                if int(total_images) > 0:
                    keep_going -= 1
                else:
                    upper_bound = upper_bound + timeskip

            except KeyboardInterrupt:
                print('Keyboard exception while querying for images, exiting\n')
                raise
            except Exception as e:
                print(type(e), e)
                print('Exception encountered while querying for images\n')

        print('finished binary search')
        s = '\nmintime: ' + str(mintime) + ' maxtime: ' + str(maxtime)
        print(s)
        out_file.write(s + '\n')

        i = getattr(rsp, 'photos', None)
        if i:
            s = 'numimgs: ' + total_images
            print(s)
            out_file.write(s + '\n')

            num_pages = int(rsp.photos[0]['pages'])
            s = 'total pages: ' + str(num_pages)
            print(s)
            out_file.write(s + '\n')

            num_visit_pages = min(16, num_pages)
            s = 'visiting only ' + str(num_visit_pages) + ' pages ( up to ' + str(num_visit_pages * 250) + ' images)'
            print(s)
            out_file.write(s + '\n')

            total_images_queried += min((num_visit_pages * 250), int(total_images))

            pagenum = 1
            while pagenum <= num_visit_pages:
                print('  page number ' + str(pagenum))
                try:
                    rsp = fapi.photos_search(
                        api_key=flickrAPIKey,
                        ispublic="1",
                        media="photos",
                        per_page="250",
                        page=str(pagenum),
                        sort="interestingness-desc",
                        has_geo="1",
                        text=query_string,
                        accuracy="6",
                        extras="tags, original_format, license, geo, date_taken, date_upload, o_dims, views",
                        min_upload_date=str(mintime),
                        max_upload_date=str(maxtime)
                    )
                    time.sleep(1)
                    FlickrAPI.testFailure(rsp)
                except KeyboardInterrupt:
                    print('Keyboard exception while querying for images, exiting\n')
                    raise
                except Exception as e:
                    print(type(e), e)
                    print('Exception encountered while querying for images\n')
                else:
                    k = getattr(rsp, 'photos', None)
                    if k:
                        m = getattr(rsp.photos[0], 'photo', None)
                        if m:
                            current_image_num = 1
                            for b in rsp.photos[0].photo:
                                if b is None:
                                    continue
                                out_file.write('photo: ' + b['id'] + ' ' + b['secret'] + ' ' + b['server'] + '\n')
                                out_file.write('owner: ' + b['owner'] + '\n')
                                out_file.write('title: ' + str(b['title']) + '\n')

                                out_file.write('originalsecret: ' + b['originalsecret'] + '\n')
                                out_file.write('originalformat: ' + b['originalformat'] + '\n')
                                out_file.write('o_height: ' + b['o_height'] + '\n')
                                out_file.write('o_width: ' + b['o_width'] + '\n')
                                out_file.write('datetaken: ' + str(b['datetaken']) + '\n')
                                out_file.write('dateupload: ' + str(b['dateupload']) + '\n')

                                out_file.write('tags: ' + str(b['tags']) + '\n')

                                out_file.write('license: ' + str(b['license']) + '\n')
                                out_file.write('latitude: ' + str(b['latitude']) + '\n')
                                out_file.write('longitude: ' + str(b['longitude']) + '\n')
                                out_file.write('accuracy: ' + str(b['accuracy']) + '\n')

                                out_file.write('views: ' + str(b['views']) + '\n')
                                out_file.write('interestingness: ' + str(current_image_num) +
                                               ' out of ' + str(total_images) + '\n\n')
                                current_image_num += 1
                pagenum += 1

            timeskip = maxtime - mintime
            mintime = maxtime

    out_file.write('Total images queried: ' + str(total_images_queried) + '\n')
    out_file.close()
