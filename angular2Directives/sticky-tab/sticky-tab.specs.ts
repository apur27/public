import {
  beforeEachProviders,
  describe
} from '@angular/core/testing';

// Load the implementations that should be tested
import { StickyTabDirective } from './sticky-tab.directive';
import { DOCUMENT } from '@angular/platform-browser';
class MockDocument {
  public body: any = {
    attributes: { 'class': { value: 'body' } }
  };
}
describe('Sticky Tab - ', () => {
  // provide our implementations or mocks to the dependency injector
  beforeEachProviders(() => [
    StickyTabDirective,
    { provide: DOCUMENT, useClass: MockDocument }
  ]);
  /*
      * Globally referenced
      * TODO: Add the test case to check sticky tab class is part of document.body.classList
      * - when the wrapper for Windows is available in Angular 2
      */
  //it('should add the sticky class on scroll event', inject([StickyTabDirective, DOCUMENT],
  //  (stickytabdirective, document) => {
  //  stickytabdirective.track();
  //  expect(document.body.classList).toEqual('sticky-tabs');
  //}));

});
