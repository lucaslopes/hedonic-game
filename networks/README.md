# Network Graphs (Datasets)

## Sample Graphs
> Simple Network Graphs

There are 2 graphs:

1. Sample **1**

![Graph 1](details/imgs/sample-1.png)

2. Sample **2**

![Graph 2](details/imgs/sample-2.png)


<!--------------------------------------------------------------->


## Terrorist Network
> From this [paper](https://journals.uic.edu/ojs/index.php/fm/rt/printerFriendly/941/863)

Network from [Figure 4](https://journals.uic.edu/ojs/index.php/fm/rt/printerFriendly/941/863#fig4):

![Graph](details/imgs/network-graph.jpg)

And its representation with numbers:
_(the .csv file is in numeric representation)_

![Graph](details/imgs/network-numbers.jpg)

As well in table format for better readability:

| Number | Name                              |
|--------|-----------------------------------|
| 01     | Abu Zubeida                       |
| 02     | Jean-Marc Grandvisir              |
| 03     | Nizar Trabelsi                    |
| 04     | Djamal Beghal                     |
| 05     | Abu Walid                         |
| 06     | Jerome Courtaillier               |
| 07     | Kamel Daoudi                      |
| 08     | David Courtaillier                |
| 09     | Zacarias Moussaoui                |
| 10     | Abu Qatada                        |
| 11     | Ahmed Ressam                      |
| 12     | Haydar Aby Doha                   |
| 13     | Mehdi Khammoun                    |
| 14     | Essoussi Laaroussi                |
| 15     | Tarek Maaroufi                    |
| 16     | Mohamed Bensakhria                |
| 17     | Tased Ben Heni                    |
| 18     | Seifallah Ben Hassine             |
| 19     | Essid Sami Ben Khemais            |
| 20     | Fahid al Shakri                   |
| 21     | Madjid Sahoune                    |
| 22     | Samir Kishk                       |
| 23     | Imad Eddin Barakat Yarkas         |
| 24     | Mohammed Belfas                   |
| 25     | Abdelghani Mzoudi                 |
| 26     | Ramzi Bin al-Shibh                |
| 27     | Agus Budiman                      |
| 28     | Mounir El Motassadeq              |
| 29     | Zakariya Essabar                  |
| 30     | Mohamed Atta                      |
| 31     | Ahmed Khalil Ibrahim Samir Al-Ani |
| 32     | Mustafa Ahmed al-Hisawi           |
| 33     | Fayez Ahmed                       |
| 34     | Wail Alshehri                     |
| 35     | Waleed Alshehri                   |
| 36     | Satam Suqami                      |
| 37     | Mohand Alshehri                   |
| 38     | Raed Hijazi                       |
| 39     | Nabil al-Marabh                   |
| 40     | Saeed Alghamdi                    |
| 41     | Hamza Alghamdi                    |
| 42     | Ahmed Alnami                      |
| 43     | Nawaf Alhazmi                     |
| 44     | Mohamed Abdi                      |
| 45     | Abdussattar Shaikh                |
| 46     | Osama Awadallah                   |
| 47     | Khalid Al-Mihdhar                 |
| 48     | Majed Moqed                       |
| 49     | Salem Alhazmi                     |
| 50     | Ahmed Alghamdi                    |
| 51     | Ahmed Al Haznawi                  |
| 52     | Abdul Aziz Al-Omari               |
| 53     | Marwan Al-Shehhi                  |
| 54     | Ziad Jarrah                       |
| 55     | Said Bahaji                       |
| 56     | Mamoun Darkazanli                 |
| 57     | Mamduh Mahmud Salim               |
| 58     | Lotfi Raissi                      |
| 59     | Hani Hanjour                      |
| 60     | Bandar Alhazmi                    |
| 61     | Rayed Mohammed Abdullah           |
| 62     | Faisal Al Salmi                   |


<!--------------------------------------------------------------->


## Conference Dataset
> From this [paper](https://arxiv.org/abs/1205.6233)

Nodes: 317080 Edges: 1049866

[DBLP collaboration network and ground-truth communities.](http://snap.stanford.edu/data/com-DBLP.html)

| File                         | Description                           |
|------------------------------|---------------------------------------|
| com-dblp.ungraph.txt.gz      | Undirected DBLP co-authorship network |
| com-dblp.all.cmty.txt.gz     | DBLP communities                      |
| com-dblp.top5000.cmty.txt.gz | DBLP communities (Top 5,000)          |
| conference.csv               | Network as CSV                        |
| conference-crop.csv          | A Sample from the Network             |


## Zachary karate club

name: Zachary karate club
code: ZA
url: http://vlado.fmf.uni-lj.si/pub/networks/data/ucinet/ucidata.htm#zachary
category: HumanSocial
description: Member–member ties
long-description: This is the well-known and much-used Zachary karate club network.  The data was collected from the members of a university karate club by Wayne Zachary in 1977.  Each node represents a member of the club, and each edge represents a tie between two members of the club.  The network is undirected.  An often discussed problem using this dataset is to find the two groups of people into which the karate club split after an argument between two teachers.
entity-names:  member
relationship-names: tie
extr: ucidata
cite: konect:ucidata-zachary
timeiso:  1977


----------------------------------------------------------------------------

#### Zachary karate club network, part of the Koblenz Network Collection

================================================

This directory contains the TSV and related files of the ucidata-zachary network: This is the well-known and much-used Zachary karate club network.  The data was collected from the members of a university karate club by Wayne Zachary in 1977.  Each node represents a member of the club, and each edge represents a tie between two members of the club.  The network is undirected.  An often discussed problem using this dataset is to find the two groups of people into which the karate club split after an argument between two teachers.


More information about the network is provided here:
http://konect.cc/networks/ucidata-zachary

Files:
    meta.ucidata-zachary -- Metadata about the network
    out.ucidata-zachary -- The adjacency matrix of the network in whitespace-separated values format, with one edge per line
      The meaning of the columns in out.ucidata-zachary are:
        First column: ID of from node
        Second column: ID of to node
        Third column (if present): weight or multiplicity of edge
        Fourth column (if present):  timestamp of edges Unix time


Use the following References for citation:

@MISC{konect:2017:ucidata-zachary,
    title = {Zachary karate club network dataset -- {KONECT}},
    month = oct,
    year = {2017},
    url = {http://konect.cc/networks/ucidata-zachary}
}

@article{konect:ucidata-zachary,
	author = {Wayne Zachary},
	year = {1977},
	title = {An Information Flow Model for Conflict and Fission in Small Groups},
	journal = {J. of Anthropol. Res.},
	volume = {33},
	pages = {452--473},
}

@article{konect:ucidata-zachary,
	author = {Wayne Zachary},
	year = {1977},
	title = {An Information Flow Model for Conflict and Fission in Small Groups},
	journal = {J. of Anthropol. Res.},
	volume = {33},
	pages = {452--473},
}


@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}

@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://dl.acm.org/citation.cfm?id=2488173},
	url_presentation = {https://www.slideshare.net/kunegis/presentationwow},
	url_web = {http://konect.cc/},
	url_citations = {https://scholar.google.com/scholar?cites=7174338004474749050},
}


## Football

https://spiral.imperial.ac.uk/handle/10044/1/30134

## Dolphins

http://www.casos.cs.cmu.edu/computational_tools/datasets/external/dolphins/index11.php
