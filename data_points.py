# This file has all the data points dictionary that are hardcoded specifically
# for the office map image. This is there so that we can impose constraints on the
# simulation when the agents are interacting with walls obstacles or end points
office_left_x = 41
office_right_x = 949
office_bottom_y = 844
office_top_y1 = 138
office_right_x1 = 439
office_top_y2 = 43



office_corner_cordinate_dict = { 1 : [(42 , 470), (440 , 472),(42 , 138) , (440 , 138)],
                                 2 : [(42 , 848) , (292 , 848) ,(42 , 564),(292 , 564)],
                                 3 : [(292 , 848),(482 , 848),(292 , 564),(482 , 564)],
                                 4 : [(482, 848),(672 , 848),(482 , 564),(672 ,564)],
                                 5 : [(672,848) , (948 , 848),(672,564),(948 , 564)],
                                 6 : [(758,472) , (948,472) ,(758,357) , (948,357)],
                                 7 : [(758,357),(948,357),(758,178),(948,178)]}

wall_end_point_cordinate_dict = {1 : [(42,138) ,(42,470)],
                                 2 : [(42,138),(440,138)],
                                 3 : [(42,472),(82,472)],
                                 4 : [(145,472),(440,472)],
                                 5 : [(440,138),(440,472)],
                                 6 : [(42,564),(42,848)],
                                 7 : [(42,848),(292,848)],
                                 8 : [(292,848),(292,564)],
                                 9 : [(270,564),(292,564)],
                                 10 : [(42,564),(206,564)],
                                 11 : [(292,564),(402,564)],
                                 12 : [(464,564),(482,564)],
                                 13 : [(292,848),(482,848)],
                                 14 : [(482,564),(482,848)],
                                 15 : [(482,564),(592,564)],
                                 16 : [(652,564),(672,564)],
                                 17 : [(672,564),(672,848)],
                                 18 : [(482,848),(672,848)],
                                 19 : [(672,564),(698,564)],
                                 20 : [(760,564),(948,564)],
                                 21 : [(948,564),(948,848)],
                                 22 : [(672,848),(948,848)],
                                 23 : [(758,472),(780,472)],
                                 24 : [(840,472),(948,472)],
                                 25 : [(758,357),(758,472)],
                                 26 : [(758,357),(948,357)],
                                 27 : [(948,357),(948,472)],
                                 28 : [(758,178),(758,357)],
                                 29 : [(948,178),(948,357)],
                                 30 : [(758,178),(770,178)],
                                 31 : [(861,178),(948,178)],
                                 32 : [(948,40),(948,178)],
                                 33 : [(440,42),(948,42)],
                                 34 : [(440,42),(440,138)]}


gate_end_point_cordinate_dict = {1 : [(82,472),(142,472),0],
                                 2 : [(210,564),(268,564),1],
                                 3 : [(404,564),(464,564),2],
                                 4 : [(596,564),(652,564),3],
                                 5 : [(700,564),(758,564),4],
                                 6 : [(780,472),(838,472),5],
                                 7 : [(776,178),(856,178),6]
                                }


graph_dict = { (115,385) :[(115,518)],
              (115, 518) :[(115,385),(241,518)],
              (241,518) : [(115,518),(242,629),(435,517)],
              (242,629) : [(241 , 518)],
              (435,517) : [(241,518),(438,631),(629,518),(624,327)],
              (438,631) : [(435,517)],
              (629,518) : [(435,517),(635,631),(736,515),(624,327)],
              (635,631) : [(629,518)],
              (736,515) : [(629,518),(624,327),(736,631),(810,511)],
              (736,631) : [(736,515)],
              (810,511) : [(736,515),(813,429)],
              (813,429) : [(810,511)],
              (624,327) : [(435,517),(629,518),(736,515),(711,121)],
              (711,121) : [(624,327),(821,116)],
              (821,116) : [(711,121),(825,220)],
              (825,220) : [(821,116)]}   

closest_graph_point_wrt_office = {1 : (115,385) , 
                                  2 : (242 , 629),
                                  3 : (438,631),
                                  4 : (635 , 631),
                                  5 : (736 , 631),
                                  6 : (813 , 429),
                                  7 : (825 , 220)}

graph_point_infront_office_dict = {(115,385) : -1,
                                   (115, 518) : 1,
                                   (241,518) :  2,
                                   (242,629) : -1,
                                   (435,517) :  3,
                                   (438,631) : -1, 
                                   (629,518) :  4,
                                   (635,631) : -1,
                                   (736,515) : 5,
                                   (736,631) : -1,
                                   (810,511) : 6,
                                   (813,429) : -1,
                                   (624,327) : -1,
                                   (711,121) : -1,
                                   (821,116) : 7,
                                   (825,220) : -1}

graph_point_inside_office_dict = { (115,385) : 1,
                                   (115, 518) : -1,
                                   (241,518) : -1,
                                   (242,629) : 2,
                                   (435,517) : -1,
                                   (438,631) : 3, 
                                   (629,518) : -1,
                                   (635,631) : 4,
                                   (736,515) : -1,
                                   (736,631) : 5,
                                   (810,511) : -1,
                                   (813,429) : 6,
                                   (624,327) : -1,
                                   (711,121) : -1,
                                   (821,116) : -1,
                                   (825,220) : 7}

obstacle_list = [[(226, 382), (226, 462), (423, 462), (423, 382)], [(50, 654), (50, 722), (220, 722), (220, 654)], [(50, 724), (50, 795), (92, 795), (92, 724)], [(119, 726), (120, 765), (165, 765), (163, 725)], [(350, 155), (350, 216), (422, 216), (422, 155)], [(325, 654), (325, 708), (446, 708), (446, 654)], [(366, 716), (366, 767), (410, 767), (410, 716)], [(512, 655), (512, 709), (635, 709), (635, 655)], [(554, 716), 
(554, 764), (597, 764), (597, 716)], [(745, 654), (745, 719), (915, 719), (915, 654)], [(862, 721), (862, 796), (916, 796), (916, 721)], [(792, 724), (792, 766), (835, 766), (835, 724)], [(507, 49), (507, 111), (635, 111), (635, 
49)], [(449, 148), (449, 309), (518, 309), (518, 148)], [(576, 166), (576, 267), (617, 267), (617, 166)], [(870, 101), (870, 159), (935, 159), (935, 101)]]
