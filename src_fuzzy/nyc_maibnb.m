clear all; close all;
T = readtable('../data/AB_NYC_2019_cleaned.csv');
n_reviews = T.number_of_reviews;
reviews_month = T.reviews_per_month;
last_review = T.last_review;

%{
hist(n_reviews, 100);
hist(reviews_month, 100);
hist(last_review, 100);
%}

fis = readfis('nyc_maibnb.fis');

input = [n_reviews, reviews_month, last_review];

confidence_on_review = evalfis(fis, input);

Tnew = removevars(T,{'number_of_reviews', 'last_review', 'reviews_per_month'});
Tnew = addvars(Tnew, confidence_on_review, 'After', 'minimum_nights');

writetable(Tnew, '../data/AB_NYC_2019_fuzzy.csv');