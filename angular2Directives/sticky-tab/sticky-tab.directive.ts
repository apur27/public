import { Directive, Inject, HostListener}
from '@angular/core';
import { DOCUMENT} from '@angular/platform-browser';

/**
 * Directive to handle scroll event on the body
 * @example
 */
@Directive({
  selector: '[lb-sticky-tab]'
})
export class StickyTabDirective {
  private _documentHeaderSelector: string;
  private _pageHeaderSelector: string;
  private _stickyTabsClassName: string;
  private _stickyTabThreshold: number;
  private _documentHeaderEl: any;
  private _documentHeaderHeight: number;
  private _documentHeight: number;
  private _headerOutOfView: boolean;
  private _pageHeaderEl: any;
  private _document: any;
  private _pageHeaderBoundingBox: any;

  /**
   * constructor - Constructor for the sticky tab directive.
   * 1. Injects the document object.
   * 2. Sets the private variable with the constants
   */
  constructor(@Inject(DOCUMENT) document: any) {
    this._document = document;
    this._documentHeaderSelector = '[role="banner"]';
    this._pageHeaderSelector = '.page-container__header';
    this._stickyTabsClassName = 'sticky-tabs';
    this._stickyTabThreshold = 400;
  }
  /**
   * track - event Listener for window's scroll event.
   * Adds the sticky tab class to document body classlist
   */
  @HostListener('window:scroll', ['$event'])
  public track($event: Event): void {
    const currentScrollY: number = window.scrollY;
    // Add "sticky tabs" class to <body>, if required
    // first condition is if current scroll Y position should be less then
    // threshold
    if (currentScrollY < this._stickyTabThreshold) {
      this._documentHeaderEl = this._document.querySelector(this._documentHeaderSelector);
      this._documentHeaderHeight  = this._documentHeaderEl ? this._documentHeaderEl.offsetHeight : undefined;
      this._documentHeight        = this._document.documentElement.clientHeight;
      this._pageHeaderEl          = this._document.querySelector(this._pageHeaderSelector);
      this._pageHeaderBoundingBox = this._pageHeaderEl ? this._pageHeaderEl.getBoundingClientRect() : undefined;
      // second condition is Header height of document AND
      // page hounder boudning box are Defined
      if (this._documentHeaderHeight !== undefined && this._pageHeaderBoundingBox !== undefined) {
        this._headerOutOfView = this._pageHeaderBoundingBox.bottom <
        this._documentHeaderHeight
        && this._pageHeaderBoundingBox.bottom <= this._documentHeight;
        //third condition is if header is out of view
        // which means header bouidng box bottom is less then
        // dcoument header height AND
        // is also less then document height
        if (this._headerOutOfView) {
          this._document.body.classList.add(this._stickyTabsClassName);
        } else {
          this._document.body.classList.remove(this._stickyTabsClassName);
        }
      }
    } else {        // if scroll distance is over the threshold, don't bother checking, just add the class.
      this._document.body.classList.add(this._stickyTabsClassName);
    }

  }
}
